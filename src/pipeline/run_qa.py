import os
import sys
import argparse
import torch
from typing import List, Dict, Any

# Add parent directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa.rule_based_qa import RuleBasedQA
from qa.llm_qa import LocalLLMQuestionAnswering
from qa.rag_qa import RAGBookQASystem

class QAPipeline:
    def __init__(
        self, 
        data_path: str = "data/books_data_20250202_163005.csv",
        method: str = "rule_based"
    ):
        """
        Initialize QA Pipeline with multiple question answering methods.
        
        Args:
            data_path (str): Path to the book data CSV
            method (str): QA method to use ('rule_based', 'unsupervised', 'supervised')
        """
        self.data_path = data_path
        self.method = method
        
        # Initialize QA systems
        self.rule_based_qa = RuleBasedQA(data_path)
        self.llm_qa = LocalLLMQuestionAnswering(data_path)
        
        # Cached models to avoid repeated training
        self.unsupervised_model = None
        self.unsupervised_tokenizer = None
        self.supervised_model = None
        self.supervised_tokenizer = None
    
    def _train_models(self):
        """
        Lazy training of LLM models if not already trained.
        """
        if self.unsupervised_model is None:
            print("Training Unsupervised Model...")
            self.unsupervised_model, self.unsupervised_tokenizer = \
                self.llm_qa.train_unsupervised(num_epochs=2)
        
        if self.supervised_model is None:
            print("Training Supervised Model...")
            self.supervised_model, self.supervised_tokenizer = \
                self.llm_qa.train_supervised(num_epochs=2)
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answer a query using the specified method.
        
        Args:
            query (str): User's natural language query
        
        Returns:
            Dict with answer and method details
        """
        results = {
            "query": query,
            "method": self.method
        }
        
        try:
            if self.method == "rule_based":
                results["answer"] = self.rule_based_qa.run_query(query)
            
            elif self.method == "unsupervised":
                self._train_models()
                results["answer"] = self.llm_qa.generate_answer(
                    query, 
                    self.unsupervised_model, 
                    self.unsupervised_tokenizer
                )
            
            elif self.method == "supervised":
                self._train_models()
                results["answer"] = self.llm_qa.generate_answer(
                    query, 
                    self.supervised_model, 
                    self.supervised_tokenizer
                )
            
            elif self.method == "rag":
                results = self.rag_qa.answer_query(query)
            
            else:
                raise ValueError(f"Invalid method: {self.method}")
            
            results["success"] = True
        
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in sequence.
        
        Args:
            queries (List[str]): List of user queries
        
        Returns:
            List of query results
        """
        return [self.answer_query(query) for query in queries]

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Book QA Pipeline")
    parser.add_argument(
        "--method", 
        choices=["rule_based", "unsupervised", "supervised"], 
        default="rule_based",
        help="QA method to use"
    )
    parser.add_argument(
        "--data", 
        default="data/books_data_20250202_163005.csv", 
        help="Path to book data CSV"
    )
    parser.add_argument(
        "--query", 
        help="Single query to process"
    )
    parser.add_argument(
        "--query-file", 
        help="File with queries (one per line)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    qa_pipeline = QAPipeline(
        data_path=args.data, 
        method=args.method
    )
    
    # Process queries
    if args.query:
        result = qa_pipeline.answer_query(args.query)
        print(f"Query: {result['query']}")
        print(f"Method: {result['method']}")
        print(f"Answer: {result.get('answer', 'No answer')}")
    
    elif args.query_file:
        with open(args.query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        results = qa_pipeline.batch_query(queries)
        for result in results:
            print(f"Query: {result['query']}")
            print(f"Method: {result['method']}")
            print(f"Answer: {result.get('answer', 'No answer')}\n")

if __name__ == "__main__":
    main()


# # Single query with rule-based method
# python run_qa.py --query "Are there books in the Horror category priced below £90?"

# # Batch queries with unsupervised LLM
# python run_qa.py --method unsupervised --query-file queries.txt

