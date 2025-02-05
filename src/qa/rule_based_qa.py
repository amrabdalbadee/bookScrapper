import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import multiprocessing
import queue
from src.processing.data_handler import BookDataHandler

import pandas as pd
import numpy as np
import multiprocessing
from transformers import pipeline, AutoModelForSeq2SeqLM
import torch
import re
from typing import Dict, Any
import os
import transformers

import warnings
warnings.simplefilter("ignore")


# Specify a local cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "huggingface_cache")

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

class RuleBasedQA:
    
    def __init__(self, data_file: str) -> None:
        """Initialize the Rule-Based QA system.

        Args:
            data_file (str): Path to the CSV file containing book data.
        """
        self.data_handler = BookDataHandler()
        self.data = self.data_handler.load_csv(data_file)
        # Load Hugging Face model for code generation
        # Model sizes:
        # 9GB: "mistralai/Mistral-7B-Instruct-v0.1"
        # 3GB: "google/flan-t5-large" (currently used)
        # 1GB: "google/flan-t5-base" 
        # "codellama/CodeLlama-7b-Python-hf"
        modelname = "codellama/CodeLlama-7b-Python-hf"
        # Load model and tokenizer with local caching
        # model = AutoModelForSeq2SeqLM.from_pretrained(
        #     modelname, 
        #     cache_dir=CACHE_DIR
        # )
        tokenizer = AutoTokenizer.from_pretrained(
            modelname, 
            cache_dir=CACHE_DIR
        )

        pipeline = transformers.pipeline(
            "text-generation",
            model=modelname,
            torch_dtype=torch.float16,
            device_map="auto",
            )

        sequences = pipeline(
            'import socket\n\ndef ping_exponential_backoff(host: str):',
            do_sample=True,
            top_k=10,
            temperature=0.1,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
            truncation=True
            )


        # self.generator = pipeline("text2text-generation", 
        #                           model=model,tokenizer=tokenizer,
        #                           device=0 if torch.cuda.is_available() else -1,
        #                           max_length=256,  # Limit response length
        #                           repetition_penalty=1.2,  # Reduce repetition
        #                           temperature=0.3,  # Lower randomness
        #                           num_return_sequences=1  # One output only
        #                           )
        #self.generator = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1, cache_dir=CACHE_DIR)  # Specify the local cache directory

        # Query type patterns for classification
        self.query_patterns = {
            'categorical': r'(are there|does|is there|do)',
            'numerical': r'(what is|how many|calculate|count|average|mean|sum|total)',
            'comparison': r'(which|compare|highest|lowest|most|least)'
        }
       
    def classify_query_type(self, query: str) -> str:
        """Determine the type of query based on its structure.
        Args:
            query (str): The user's question
        Returns:
            str: Query type ('categorical', 'numerical', or 'comparison')
        """
        query = query.lower()
        for query_type, pattern in self.query_patterns.items():
            if re.search(pattern, query):
                return query_type
        return 'categorical'  # default type

    def generate_code(self, query: str) -> str:
        """Generate Python code to process a given query using the book dataset.
        Args:
            query (str): User's natural language query.
        Returns:
            str: The generated Python code.
        """
        query_type = self.classify_query_type(query)
        
        # Build prompt based on query type
        base_prompt = f"""
        Given a pandas DataFrame 'data' with book information, return only valid, executable Python code to answer this query: {query}
        
        The DataFrame has these columns: title, category, price, description, is_available, stock_count, star_rating
        """
        
        if query_type == 'categorical':
            base_prompt += """
            Example categorical queries and their code:
            
            Query: "Are there books in the Horror category priced below £90?"
            Code:
            filtered_data = data[(data['category'] == 'Horror') & (data['price'] < 90)]
            answer = len(filtered_data) > 0
            
            Query: "Does Mystery category have 5-star ratings?"
            Code:
            filtered_data = data[(data['category'] == 'Mystery') & (data['star_rating'] == 5)]
            answer = len(filtered_data) > 0
            """
        
        elif query_type == 'numerical':
            base_prompt += """
            Example numerical queries and their code:
            
            Query: "What is the average price of books in each category?"
            Code:
            answer = data.groupby('category')['price'].mean().round(2).to_dict()
            
            Query: "How many books are available in stock?"
            Code:
            answer = len(data[data['is_available'] == 1)
            """
        
        else:  # comparison
            base_prompt += """
            Example comparison queries and their code:
            
            Query: "Which category has the highest average price?"
            Code:
            avg_prices = data.groupby('category')['price'].mean()
            answer = avg_prices.idxmax()
            
            Query: "Which categories have more than 50% books above £30?"
            Code:
            total_books = data.groupby('category').size()
            expensive_books = data[data['price'] > 30].groupby('category').size()
            percentages = (expensive_books / total_books * 100).round(2)
            answer = percentages[percentages > 50].index.tolist()
            """
        
        base_prompt += """
        Ensure the code:
        1. Uses pandas operations.
        2. Returns only a Python function.
        3. Does not use loops unless necessary.
        4. Stores the result in a variable named 'answer'.
        5. Avoids arbitrary assignments and nonsensical loops.
        """
        
        #base_prompt = "Introduce your self"
        response = self.generator(
            base_prompt,
            max_length=500,
            num_return_sequences=1
        )[0]['generated_text']
       
        print("response is \n")
        print(response)

        if not self.validate_code(response):
            response = "answer = None  # Model failed to generate valid code"
                
        return response
    
    # def generate_code(self, query: str) -> str:
    #     """Generate Python code to process a given query using the book dataset.

    #     Args:
    #         query (str): User's natural language query.

    #     Returns:
    #         str: The generated Python code.
    #     """
    #     prompt = f"""
    #     Given the following structured book dataset:
        
    #     Sample Data:
    #     {self.data[:2]}  # Providing a subset of the dataset
        
    #     Methods:
    #     - Price filtering (e.g., books below a certain price)
    #     - Availability checks (e.g., out of stock books)
    #     - Rating-based filtering (e.g., books with a 5-star rating)
    #     - Category-based analysis (e.g., number of books per category)
        
    #     Query: {query}
        
    #     Generate a Python function that processes the given dataset to answer the query.
    #     Ensure the function follows best practices and is directly executable.
    #     """
        
    #     response = self.generator(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
    #     return response

    def validate_code(self, code):
        try:
            parsed = ast.parse(code)
            return any(node.id == "answer" for node in ast.walk(parsed))
        except SyntaxError:
            return False
    
    def execute_code(self, code: str, query: str) -> str:
        """Execute the generated Python code to answer the query.
        Args:
            code (str): Python code to be executed.
            query (str): Original query from the user.
        Returns:
            str: The result of executing the generated code.
        """
        namespace = {
            "data": self.data,
            "pd": pd,
            "np": np
        }
        
        try:
            exec(code, namespace)
            result = namespace.get("answer", None)
            
            if result is None:
                return "Could not determine an answer"
                
            # Format the result based on its type
            if isinstance(result, bool):
                return "Yes" if result else "No"
            elif isinstance(result, (int, float)):
                return f"{result:,.2f}" if isinstance(result, float) else str(result)
            elif isinstance(result, dict):
                return "\n".join([f"{k}: {v:,.2f}" for k, v in result.items()])
            elif isinstance(result, list):
                return ", ".join(map(str, result))
            else:
                return str(result)
                
        except Exception as e:
            return f"Error executing generated code: {str(e)}"
    
    def process_query(self, query: str, queue: multiprocessing.Queue) -> None:
        """Generate and execute code for the given query, then store the result in a queue.
        Args:
            query (str): The user's query.
            queue (multiprocessing.Queue): A queue to store the result.
        """
        generated_code = self.generate_code(query)
        print("Generated code:")
        print(generated_code)
        result = self.execute_code(generated_code, query)
        queue.put(result)

    def run_query(self, query: str) -> str:
        """Run a query in a separate process to handle execution safely.
        Args:
            query (str): The user's query.
        Returns:
            str: The result of the executed query.
        """
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self.process_query,
            args=(query, queue)
        )
        process.start()
        process.join(timeout=120)  # Add timeout
        
        if process.is_alive():
            process.terminate()
            return "Query timed out"
            
        return queue.get()

if __name__ == "__main__":
    qa_system = RuleBasedQA("data/books_data_20250202_163005.csv")
    
    # Test different types of queries
    test_queries = [
        # Categorical
        "Are there books in the 'Travel' category that are marked as 'Out of stock'?",
        "Does the 'Mystery' category contain books with a 5-star rating?",
        
        # Numerical
        "What is the average price of books across each category?",
        "What is the price range for books in the 'Historical Fiction' category?",
        
        # Hybrid/Comparison
        "Which category has the highest average price of books?",
        "Which categories have more than 50% of their books priced above £30?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = qa_system.run_query(query)
        print(f"Answer: {result}")


