import google.generativeai as genai
import pandas as pd
import numpy as np
import ast
import re
import os
import signal
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import time
from src.processing.data_handler import BookDataHandler
import warnings
warnings.simplefilter("ignore")

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout handling."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Set the signal handler and a timeout
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

class GeminiQA:
    def __init__(self, data_file: str, api_key: str = None) -> None:
        """Initialize the Gemini-powered QA system.

        Args:
            data_file (str): Path to the CSV file containing book data.
            api_key (str, optional): Google Cloud API key for Gemini.
        """
        self.data_handler = BookDataHandler()
        self.data = self.data_handler.load_csv(data_file)
        print("columns Datatypes are \n")
        # Configure Gemini - try different methods of getting API key
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided. Please provide an API key either through the constructor or GOOGLE_API_KEY environment variable.")
        
        # Configure Gemini with explicit API key
        genai.configure(api_key=api_key)
        
        try:
            # Test the API key by creating the model
            self.model = genai.GenerativeModel('gemini-pro')
            # Try a simple generation to verify the API key works
            self.model.generate_content("Test")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini model. Error: {str(e)}")
        
        # Query type patterns for classification
        self.query_patterns = {
            'categorical': r'(are there|does|is there|do)',
            'numerical': r'(what is|how many|calculate|count|average|mean|sum|total)',
            'comparison': r'(which|compare|highest|lowest|most|least)'
        }

    def classify_query_type(self, query: str) -> str:
        """Determine the type of query based on its structure."""
        query = query.lower()
        for query_type, pattern in self.query_patterns.items():
            if re.search(pattern, query):
                return query_type
        return 'categorical'  # default type

    def generate_code(self, query: str) -> str:
        """Generate Python code using Gemini to process a given query."""
        query_type = self.classify_query_type(query)
        
        base_prompt = f"""
        You are a Python code generator. Given a pandas DataFrame 'data' with book information, 
        generate only valid, executable Python code to answer this query: {query}
        
        The DataFrame has these columns: title, category, price, description, is_available, stock_count, star_rating
        
        Requirements:
        1. Use pandas operations
        2. Store the final result in a variable named 'answer'
        3. Avoid loops unless absolutely necessary
        4. Return only the Python code, no explanations
        5. The code must be complete and executable
        
        Example for {query_type} query:
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

        try:
            response = self.model.generate_content(base_prompt)
            generated_code = self.clean_generated_code(response.text)
            
            if not self.validate_code(generated_code):
                return "answer = None  # Model failed to generate valid code"
            
            return generated_code
            
        except Exception as e:
            print(f"Error generating code: {str(e)}")
            return "answer = None  # Error in code generation"
    
    def clean_generated_code(self, code: str) -> str:
        """Clean up the generated code by removing markdown and unnecessary text."""
        code = re.sub(r'```python\n|```\n|```', '', code)
        code_lines = [line for line in code.strip().split('\n')
                     if line.strip() and not line.strip().startswith('#')]
        return '\n'.join(code_lines)

    def validate_code(self, code: str) -> bool:
        """Validate the generated code."""
        try:
            parsed = ast.parse(code)
            return any(isinstance(node, ast.Assign) and 
                      any(t.id == "answer" for t in node.targets if isinstance(t, ast.Name))
                      for node in ast.walk(parsed))
        except SyntaxError:
            return False
    
    def execute_code(self, code: str) -> tuple:
        """Execute the generated Python code and return both answer and explanation."""
        namespace = {
            "data": self.data,
            "pd": pd,
            "np": np
        }
        
        try:
            exec(code, namespace)
            result = namespace.get("answer", None)
            
            if result is None:
                return "Could not determine an answer", "The generated code did not return a valid result."
            
            # Generate explanation based on result type
            explanation = "This result is based on querying the book dataset using pandas operations."
            
            if isinstance(result, bool):
                explanation += " The response is 'Yes' if there are matching records and 'No' otherwise."
                result = "Yes" if result else "No"
            elif isinstance(result, (int, float)):
                explanation += " The result represents a numerical calculation, such as a count or an average price."
                result = f"{result:,.2f}" if isinstance(result, float) else str(result)
            elif isinstance(result, dict):
                explanation += " This dictionary represents values grouped by category."
                result = "\n".join([f"{k}: {v:,.2f}" for k, v in result.items()])
            elif isinstance(result, list):
                explanation += " The list contains categories that satisfy the given conditions."
                result = ", ".join(map(str, result))
            else:
                result = str(result)
            
            return result, explanation
        
        except Exception as e:
            return "Error executing generated code", f"An error occurred: {str(e)}"


    def run_query(self, query: str, timeout_seconds: int = 120) -> tuple:
        """Run a query with timeout handling and return both answer and explanation."""
        try:
            with timeout(timeout_seconds):
                generated_code = self.generate_code(query)
                print("Generated code:")
                print(generated_code)
                answer, explanation = self.execute_code(generated_code)
                return answer, explanation
        except TimeoutException:
            return "Query timed out", "The query took too long to execute and was aborted."
        except Exception as e:
            return f"Error processing query: {str(e)}", "An unexpected error occurred while processing the query."

if __name__ == "__main__":
    # Set your API key here
    API_KEY = "AIzaSyBYWae9t_15c135GITaDRR5UOYyj0OliT0"
    
    try:
        qa_system = GeminiQA("data/books_data_20250202_163005.csv", API_KEY)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure you have set the correct API key.")
        exit(1)
    
    # Test different types of queries
    test_queries = [
        "Are there books in the 'Travel' category that are marked as 'Out of stock'?",
        "Does the 'Mystery' category contain books with a 5-star rating?",
        "What is the average price of books across each category?",
        "What is the price range for books in the 'Historical Fiction' category?",
        "Which category has the highest average price of books?",
        "Which categories have more than 50% of their books priced above £30?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        answer, explanation = qa_system.run_query(query)
        print(f"Answer: {answer}")
        print(f"Explanation: {explanation}")