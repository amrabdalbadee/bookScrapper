import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.processing.data_handler import BookDataHandler

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Transformers and ML libraries
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sklearn.metrics.pairwise import cosine_similarity

class RAGBookQASystem:
    def __init__(
        self, 
        books_data: str = "data/books_data_20250202_163005.csv",
        embedding_model: str = "all-MiniLM-L6-v2",
        qa_model: str = "deepset/roberta-base-squad2"
    ):
        """
        Initialize RAG-based Book QA System
        
        Args:
            data_path (str): Path to book dataset CSV
            embedding_model (str): Sentence transformer for embeddings
            qa_model (str): Question answering model
        """
        # Load book data
        self.data_handler = BookDataHandler()
        self.data = self.data_handler.load_csv(books_data)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Precompute book embeddings
        self.book_embeddings = self._compute_book_embeddings()
        
        # Initialize QA pipeline
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=qa_model,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def _compute_book_embeddings(self) -> np.ndarray:
        """
        Compute embeddings for books using concatenated text.
        
        Returns:
            np.ndarray: Embeddings for each book
        """
        book_texts = self.data.apply(
            lambda row: f"{row['title']} {row['category']} {row['description']}", 
            axis=1
        ).tolist()
        
        return self.embedding_model.encode(book_texts)
    
    def _semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Perform semantic search to find most relevant books.
        
        Args:
            query (str): User's natural language query
            top_k (int): Number of top results to return
        
        Returns:
            List of top relevant book entries
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Compute cosine similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            self.book_embeddings
        )[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [self.data.iloc[idx].to_dict() for idx in top_indices]
    
    def _answer_categorical_question(
        self, 
        query: str, 
        relevant_books: List[Dict]
    ) -> Dict[str, Any]:
        """
        Answer categorical yes/no or count-based questions.
        
        Args:
            query (str): User's query
            relevant_books (List[Dict]): Top relevant books
        
        Returns:
            Dict with answer and justification
        """
        lowered_query = query.lower()
        
        # Categorical checks
        if "out of stock" in lowered_query:
            out_of_stock = [
                book for book in relevant_books 
                if book['is_available'] == 0
            ]
            return {
                "answer": "Yes" if out_of_stock else "No",
                "justification": f"{len(out_of_stock)} out of {len(relevant_books)} books are out of stock."
            }
        
        if "more than 50%" in lowered_query and "priced above" in lowered_query:
            price_threshold = float(query.split("£")[1].split()[0])
            high_priced = [
                book for book in relevant_books 
                if book['price'] > price_threshold
            ]
            return {
                "answer": "Yes" if len(high_priced) > len(relevant_books) / 2 else "No",
                "justification": f"{len(high_priced)} out of {len(relevant_books)} books are above £{price_threshold}"
            }
        
        return {"answer": "Unable to determine", "justification": "Query too complex"}
    
    def _answer_numerical_question(
        self, 
        query: str, 
        relevant_books: List[Dict]
    ) -> Dict[str, Any]:
        """
        Answer numerical questions about books.
        
        Args:
            query (str): User's query
            relevant_books (List[Dict]): Top relevant books
        
        Returns:
            Dict with answer and justification
        """
        lowered_query = query.lower()
        
        # Numerical analyses
        if "average price" in lowered_query:
            avg_price = np.mean([book['price'] for book in relevant_books])
            return {
                "answer": f"£{avg_price:.2f}",
                "justification": f"Calculated from {len(relevant_books)} books"
            }
        
        if "price range" in lowered_query:
            prices = [book['price'] for book in relevant_books]
            return {
                "answer": f"£{min(prices):.2f} - £{max(prices):.2f}",
                "justification": f"Range from {len(relevant_books)} books"
            }
        
        if "books in stock" in lowered_query:
            in_stock = sum(book['is_available'] for book in relevant_books)
            return {
                "answer": str(in_stock),
                "justification": f"Out of {len(relevant_books)} books"
            }
        
        return {"answer": "Unable to determine", "justification": "Query too complex"}
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to answer book-related queries using RAG.
        
        Args:
            query (str): User's natural language query
        
        Returns:
            Dict with query results
        """
        # Semantic search for relevant books
        relevant_books = self._semantic_search(query)
        
        # Try categorical questions first
        categorical_answer = self._answer_categorical_question(query, relevant_books)
        if categorical_answer["answer"] != "Unable to determine":
            return {
                "method": "RAG-Categorical",
                "answer": categorical_answer["answer"],
                "justification": categorical_answer["justification"]
            }
        
        # Try numerical questions
        numerical_answer = self._answer_numerical_question(query, relevant_books)
        if numerical_answer["answer"] != "Unable to determine":
            return {
                "method": "RAG-Numerical",
                "answer": numerical_answer["answer"],
                "justification": numerical_answer["justification"]
            }
        
        # Fallback to context-based QA
        contexts = [
            f"Title: {book['title']}, Category: {book['category']}, "
            f"Price: £{book['price']}, Available: {book['is_available']}" 
            for book in relevant_books
        ]
        
        qa_result = self.qa_pipeline({
            'question': query,
            'context': " ".join(contexts)
        })
        
        return {
            "method": "RAG-QA-Pipeline",
            "answer": qa_result['answer'],
            "score": qa_result['score']
        }

def main():
    # Initialize RAG QA System
    rag_qa = RAGBookQASystem()
    
    # Test queries
    test_queries = [
        "Are there books in the Travel category that are out of stock?",
        "What is the average price of books?",
        "What is the price range of books?",
        "How many books are in stock?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag_qa.answer_query(query)
        print("Method:", result.get('method', 'N/A'))
        print("Answer:", result.get('answer', 'N/A'))
        print("Justification:", result.get('justification', 'N/A'))

if __name__ == "__main__":
    main()