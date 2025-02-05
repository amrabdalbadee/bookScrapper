import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.processing.data_handler import BookDataHandler
import re

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Transformers and ML libraries
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.simplefilter("ignore")

# class RAGBookQASystem:
#     def __init__(
#         self, 
#         books_data: str = "data/books_data_20250202_163005.csv",
#         embedding_model: str = "all-MiniLM-L6-v2",
#         qa_model: str = "deepset/roberta-base-squad2"
#     ):
#         """
#         Initialize RAG-based Book QA System
        
#         Args:
#             data_path (str): Path to book dataset CSV
#             embedding_model (str): Sentence transformer for embeddings
#             qa_model (str): Question answering model
#         """
#         # Load book data
#         self.data_handler = BookDataHandler()
#         self.data = self.data_handler.load_csv(books_data)
        
#         # Initialize embedding model
#         self.embedding_model = SentenceTransformer(embedding_model)
        
#         # Precompute book embeddings
#         self.book_embeddings = self._compute_book_embeddings()
        
#         # Initialize QA pipeline
#         self.qa_pipeline = pipeline(
#             "question-answering", 
#             model=qa_model,
#             device=0 if torch.cuda.is_available() else -1
#         )
    
#     def _compute_book_embeddings(self) -> np.ndarray:
#         """
#         Compute embeddings for books using concatenated text.
        
#         Returns:
#             np.ndarray: Embeddings for each book
#         """
#         book_texts = self.data.apply(
#             lambda row: f"{row['title']} {row['category']} {row['description']}", 
#             axis=1
#         ).tolist()
        
#         return self.embedding_model.encode(book_texts)
    
#     def _semantic_search(self, query: str, top_k: int = None) -> pd.DataFrame:
#         """
#         Perform semantic search to find most relevant books.
        
#         Args:
#             query (str): User's natural language query
#             top_k (int, optional): Number of top results to return
        
#         Returns:
#             pandas.DataFrame of relevant books
#         """
#         # Check if query mentions a specific category
#         categories = self.data['category'].unique()
#         matching_category = next((cat for cat in categories if cat.lower() in query.lower()), None)
        
#         # If a specific category is mentioned, filter data
#         if matching_category:
#             filtered_data = self.data[self.data['category'] == matching_category]
            
#             # Embed filtered data
#             book_texts = filtered_data.apply(
#                 lambda row: f"{row['title']} {row['category']} {row['description']}", 
#                 axis=1
#             ).tolist()
            
#             query_embedding = self.embedding_model.encode([query])[0]
#             book_embeddings = self.embedding_model.encode(book_texts)
            
#             # Compute cosine similarities
#             similarities = cosine_similarity(
#                 query_embedding.reshape(1, -1), 
#                 book_embeddings
#             )[0]
            
#             # Get top-k indices
#             top_k = top_k or len(filtered_data)
#             top_indices = similarities.argsort()[-top_k:][::-1]
            
#             return filtered_data.iloc[top_indices]
        
#         # If no category specified, use original method
#         query_embedding = self.embedding_model.encode([query])[0]
#         top_k = top_k or 3
        
#         # Compute cosine similarities
#         similarities = cosine_similarity(
#             query_embedding.reshape(1, -1), 
#             self.book_embeddings
#         )[0]
        
#         # Get top-k indices
#         top_indices = similarities.argsort()[-top_k:][::-1]
        
#         return self.data.iloc[top_indices]

#     def _answer_numerical_question(self, query: str, relevant_books: pd.DataFrame) -> Dict[str, Any]:
#         """
#         Answer numerical questions about books.
        
#         Args:
#             query (str): User's query
#             relevant_books (pd.DataFrame): Top relevant books
        
#         Returns:
#             Dict with answer and justification
#         """
#         lowered_query = query.lower()
        
#         # Books available in stock across categories
#         if "books are available in stock" in lowered_query:
#             total_in_stock = self.data['is_available'].sum()
#             return {
#                 "answer": str(total_in_stock),
#                 "justification": f"Total books in stock across all {len(self.data['category'].unique())} categories"
#             }
        
#         # Average price across categories
#         if "average price" in lowered_query and "across" in lowered_query:
#             # Group by category and calculate average price
#             category_avg_prices = self.data.groupby('category')['price'].mean()
            
#             result = "\n".join([f"{cat}: £{price:.2f}" for cat, price in category_avg_prices.items()])
#             return {
#                 "answer": result,
#                 "justification": f"Average prices for {len(category_avg_prices)} categories"
#             }
        
#         # Price range
#         if "price range" in lowered_query:
#             min_price = self.data['price'].min()
#             max_price = self.data['price'].max()
#             return {
#                 "answer": f"£{min_price:.2f} - £{max_price:.2f}",
#                 "justification": f"Range from all {len(self.data)} books"
#             }
        
#         return {"answer": "Unable to determine", "justification": "Query too complex"}

#     def _answer_categorical_question(self, query: str, relevant_books: pd.DataFrame) -> Dict[str, Any]:
#         """
#         Answer categorical yes/no or count-based questions.
        
#         Args:
#             query (str): User's query
#             relevant_books (pd.DataFrame): Top relevant books
        
#         Returns:
#             Dict with answer and justification
#         """
#         lowered_query = query.lower()
        
#         # Highest percentage of out-of-stock books
#         if "highest percentage of books marked as out of stock" in lowered_query:
#             # Calculate out-of-stock percentage per category
#             out_of_stock_pct = self.data.groupby('category').apply(
#                 lambda x: (x['is_available'] == 0).mean() * 100
#             )
            
#             # Find category/categories with highest percentage
#             max_pct = out_of_stock_pct.max()
#             top_categories = out_of_stock_pct[out_of_stock_pct == max_pct].index.tolist()
            
#             return {
#                 "answer": ", ".join(top_categories),
#                 "justification": f"Highest out-of-stock percentage: {max_pct:.2f}%"
#             }
        
#         # Categories with more than 50% of books priced above threshold
#         if "more than 50% of their books priced above" in lowered_query:
#             try:
#                 price_threshold = float(re.findall(r'£(\d+)', query)[0])
                
#                 # Calculate percentage of books above price threshold per category
#                 above_price_pct = self.data.groupby('category').apply(
#                     lambda x: (x['price'] > price_threshold).mean() * 100
#                 )
                
#                 # Find categories with more than 50% of books above threshold
#                 high_price_categories = above_price_pct[above_price_pct > 50].index.tolist()
                
#                 return {
#                     "answer": ", ".join(high_price_categories) if high_price_categories else "None",
#                     "justification": "Categories with >50% books above £" + str(price_threshold)
#                 }
#             except (IndexError, ValueError):
#                 return {"answer": "Unable to determine", "justification": "Could not parse price threshold"}
        
#         # Existing categorical checks remain the same
        
#         return {"answer": "Unable to determine", "justification": "Query too complex"}
    
#     def answer_query(self, query: str) -> Dict[str, Any]:
#         """
#         Main method to answer book-related queries using RAG.
        
#         Args:
#             query (str): User's natural language query
        
#         Returns:
#             Dict with query results
#         """
#         # Try to extract category from query
#         categories = self.data['category'].unique()
#         matching_category = next((cat for cat in categories if cat.lower() in query.lower()), None)
        
#         # Filter data by category if specified
#         data_subset = self.data[self.data['category'] == matching_category] if matching_category else self.data
        
#         # Specific categorical checks with category context
#         if "5-star rating" in query.lower():
#             five_star_books = data_subset[data_subset['star_rating'] == 5]
#             return {
#                 "method": "RAG-Categorical",
#                 "answer": "Yes" if len(five_star_books) > 0 else "No",
#                 "justification": f"{len(five_star_books)} books with 5-star rating"
#             }
        
#         if "priced below" in query.lower():
#             try:
#                 price_threshold = float(re.findall(r'£(\d+)', query)[0])
#                 low_priced_books = data_subset[data_subset['price'] < price_threshold]
#                 return {
#                     "method": "RAG-Categorical",
#                     "answer": "Yes" if len(low_priced_books) > 0 else "No",
#                     "justification": f"{len(low_priced_books)} books below £{price_threshold}"
#                 }
#             except (IndexError, ValueError):
#                 pass
        
#         if "more than 50%" in query.lower() and "priced above" in query.lower():
#             try:
#                 price_threshold = float(re.findall(r'£(\d+)', query)[0])
#                 high_priced_books = data_subset[data_subset['price'] > price_threshold]
                
#                 return {
#                     "method": "RAG-Categorical",
#                     "answer": "Yes" if len(high_priced_books) > len(data_subset) / 2 else "No",
#                     "justification": f"{len(high_priced_books)} out of {len(data_subset)} books above £{price_threshold}"
#                 }
#             except (IndexError, ValueError):
#                 pass
        
#         if "out of stock" in query.lower():
#             out_of_stock_books = data_subset[data_subset['is_available'] == 0]
#             return {
#                 "method": "RAG-Categorical",
#                 "answer": "Yes" if len(out_of_stock_books) > 0 else "No",
#                 "justification": f"{len(out_of_stock_books)} books out of stock"
#             }
        
#         # Fallback to existing numerical and QA methods
#         relevant_books = self._semantic_search(query)
        
#         # Try numerical questions first
#         numerical_answer = self._answer_numerical_question(query, relevant_books)
#         if numerical_answer["answer"] != "Unable to determine":
#             return {
#                 "method": "RAG-Numerical",
#                 "answer": numerical_answer["answer"],
#                 "justification": numerical_answer["justification"]
#             }
        
#         # Fallback to QA pipeline
#         contexts = [
#             f"Title: {book['title'] if isinstance(book, dict) else book['title']}, " +
#             f"Category: {book['category'] if isinstance(book, dict) else book['category']}, " +
#             f"Price: £{book['price'] if isinstance(book, dict) else book['price']}, " +
#             f"Available: {book['is_available'] if isinstance(book, dict) else book['is_available']}" 
#             for book in (relevant_books.to_dict('records') if isinstance(relevant_books, pd.DataFrame) else relevant_books)
#         ]
        
#         qa_result = self.qa_pipeline({
#             'question': query,
#             'context': " ".join(contexts)
#         })
        
#         return {
#             "method": "RAG-QA-Pipeline",
#             "answer": qa_result['answer'],
#             "score": qa_result['score']
#         }

# def main():
#     # Initialize RAG QA System
#     rag_qa = RAGBookQASystem()
    
#      # Test questions
#     test_queries = [
#         # Categorical Questions
#         "Are there any books in the Travel category that are marked as Out of stock?",
#         "Does the Mystery category contain books with a 5-star rating?",
#         "Are there books in the Classics category priced below £10?",
#         "Are more than 50% of books in the Mystery category priced above £20?",
        
#         # Numerical Data Questions
#         "What is the average price of books across each category?",
#         "What is the price range for books in the Historical Fiction category?",
#         "How many books are available in stock across the categories?",
#         "What is the total value of books in the Travel category?",
        
#         # Hybrid Questions
#         "Which category has the highest average price of books?",
#         "Which categories have more than 50% of their books priced above £30?",
#         "Which category has the highest percentage of books marked as Out of stock?"
#     ]
    
#     for query in test_queries:
#         print(f"\nQuery: {query}")
#         result = rag_qa.answer_query(query)
#         print("Method:", result.get('method', 'N/A'))
#         print("Answer:", result.get('answer', 'N/A'))
#         print("Justification:", result.get('justification', 'N/A'))

# if __name__ == "__main__":
#     main()


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
            books_data (str): Path to book dataset CSV
            embedding_model (str): Sentence transformer for embeddings
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
    
    def _filter_by_category(self, query: str) -> pd.DataFrame:
        """
        Extract and filter data by category if mentioned in query
        
        Args:
            query (str): User's query
        
        Returns:
            pandas.DataFrame: Filtered or full dataset
        """
        categories = self.data['category'].unique()
        matching_category = next((cat for cat in categories if cat.lower() in query.lower()), None)
        
        return self.data[self.data['category'] == matching_category] if matching_category else self.data
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to answer book-related queries
        
        Args:
            query (str): User's natural language query
        
        Returns:
            Dict with query results
        """
        query_lower = query.lower()
        filtered_data = self._filter_by_category(query)

        # Books available in stock across categories
        if any(phrase in query_lower for phrase in [
            "books available in stock", 
            "how many books are available", 
            "total books in stock"
        ]):
            total_available_books = self.data[self.data['is_available'] == 1]['is_available'].count()
            return {
                "method": "Numerical",
                "answer": str(total_available_books),
                "justification": f"Total books marked as available across {len(self.data['category'].unique())} categories"
            }
        
        # Categories with more than 50% of books priced above threshold
        if "more than 50%" in query_lower and "priced above" in query_lower:
            try:
                price_threshold = float(re.findall(r'£(\d+)', query)[0])
                
                # Calculate percentage of books above price threshold per category
                category_high_price_pct = self.data.groupby('category').apply(
                    lambda x: (x['price'] > price_threshold).mean() * 100
                )
                
                # Find categories with more than 50% of books above threshold
                high_price_categories = category_high_price_pct[category_high_price_pct > 50].index.tolist()
                
                return {
                    "method": "Hybrid",
                    "answer": ", ".join(high_price_categories) if high_price_categories else "None",
                    "justification": f"Categories with >50% books above £{price_threshold}"
                }
            except (IndexError, ValueError):
                pass
        
        # Categorical (Yes/No) Questions
        if "out of stock" in query_lower:
            out_of_stock_books = filtered_data[filtered_data['is_available'] == 0]
            return {
                "method": "Categorical",
                "answer": "Yes" if len(out_of_stock_books) > 0 else "No",
                "justification": f"{len(out_of_stock_books)} books out of stock in filtered category"
            }
        
        if "5-star rating" in query_lower:
            five_star_books = filtered_data[filtered_data['star_rating'] == 5]
            return {
                "method": "Categorical",
                "answer": "Yes" if len(five_star_books) > 0 else "No",
                "justification": f"{len(five_star_books)} books with 5-star rating"
            }
        
        if "priced below" in query_lower:
            try:
                price_threshold = float(re.findall(r'£(\d+)', query)[0])
                low_priced_books = filtered_data[filtered_data['price'] < price_threshold]
                return {
                    "method": "Categorical",
                    "answer": "Yes" if len(low_priced_books) > 0 else "No",
                    "justification": f"{len(low_priced_books)} books below £{price_threshold}"
                }
            except (IndexError, ValueError):
                pass
        
        if "more than 50%" in query_lower and "priced above" in query_lower:
            try:
                price_threshold = float(re.findall(r'£(\d+)', query)[0])
                high_priced_books = filtered_data[filtered_data['price'] > price_threshold]
                
                return {
                    "method": "Categorical",
                    "answer": "Yes" if len(high_priced_books) > len(filtered_data) / 2 else "No",
                    "justification": f"{len(high_priced_books)} out of {len(filtered_data)} books above £{price_threshold}"
                }
            except (IndexError, ValueError):
                pass
        
        # Numerical Questions
        if "average price" in query_lower and "across" in query_lower:
            if filtered_data is not self.data:
                category_avg_price = filtered_data['price'].mean()
                return {
                    "method": "Numerical",
                    "answer": f"£{category_avg_price:.2f}",
                    "justification": f"Average price for {filtered_data['category'].iloc[0]} category"
                }
            else:
                category_avg_prices = self.data.groupby('category')['price'].mean()
                result = "\n".join([f"{cat}: £{price:.2f}" for cat, price in category_avg_prices.items()])
                return {
                    "method": "Numerical",
                    "answer": result,
                    "justification": f"Average prices for {len(category_avg_prices)} categories"
                }
        
        if "price range" in query_lower:
            min_price = filtered_data['price'].min()
            max_price = filtered_data['price'].max()
            return {
                "method": "Numerical",
                "answer": f"£{min_price:.2f} - £{max_price:.2f}",
                "justification": f"Price range for {len(filtered_data)} books"
            }
        
        if "books available in stock" in query_lower:
            total_in_stock = filtered_data['stock_count'].sum()
            return {
                "method": "Numerical",
                "answer": str(total_in_stock),
                "justification": f"Total books in stock for {len(filtered_data['category'].unique())} categories"
            }
        
        if "total value" in query_lower:
            total_value = filtered_data['price'].sum()
            return {
                "method": "Numerical",
                "answer": f"£{total_value:.2f}",
                "justification": f"Total value of books in the category"
            }
        
        # Hybrid/Comparative Questions
        if "highest average price" in query_lower:
            category_avg_prices = self.data.groupby('category')['price'].mean()
            highest_price_category = category_avg_prices.idxmax()
            return {
                "method": "Hybrid",
                "answer": highest_price_category,
                "justification": f"Highest average price: £{category_avg_prices.max():.2f}"
            }
        
        if "more than 50%" in query_lower and "priced above" in query_lower:
            try:
                price_threshold = float(re.findall(r'£(\d+)', query)[0])
                category_high_price_pct = self.data.groupby('category').apply(
                    lambda x: (x['price'] > price_threshold).mean() * 100
                )
                high_price_categories = category_high_price_pct[category_high_price_pct > 50].index.tolist()
                
                return {
                    "method": "Hybrid",
                    "answer": ", ".join(high_price_categories) if high_price_categories else "None",
                    "justification": f"Categories with >50% books above £{price_threshold}"
                }
            except (IndexError, ValueError):
                pass
        
        if "description length" in query_lower:
            description_lengths = self.data.groupby('category')['description'].apply(
                lambda x: x.str.split().str.len().mean()
            )
            return {
                "method": "Hybrid",
                "answer": "\n".join([f"{cat}: {length:.1f} words" for cat, length in description_lengths.items()]),
                "justification": "Average description length per category"
            }
        
        if "highest percentage of books marked as out of stock" in query_lower:
            out_of_stock_pct = self.data.groupby('category').apply(
                lambda x: (x['is_available'] == 0).mean() * 100
            )
            max_pct = out_of_stock_pct.max()
            top_categories = out_of_stock_pct[out_of_stock_pct == max_pct].index.tolist()
            
            return {
                "method": "Hybrid",
                "answer": ", ".join(top_categories),
                "justification": f"Highest out-of-stock percentage: {max_pct:.2f}%"
            }
        
        return {
            "method": "Not Supported",
            "answer": "Unable to determine",
            "justification": "Query type not recognized"
        }

def main():
    # Initialize RAG QA System
    rag_qa = RAGBookQASystem()
    
    # Test questions covering various types
    test_queries = [
        # Categorical Questions
        "Are there any books in the Travel category that are marked as Out of stock?",
        "Does the Mystery category contain books with a 5-star rating?",
        "Are there books in the Classics category priced below £10?",
        "Are more than 50% of books in the Mystery category priced above £20?",
        
        # Numerical Questions
        "What is the average price of books across each category?",
        "What is the price range for books in the Historical Fiction category?",
        "How many books are available in stock across the categories?",
        "What is the total value of books in the Travel category?",
        
        # Hybrid/Comparative Questions
        "Which category has the highest average price of books?",
        "Which categories have more than 50% of their books priced above £30?",
        "Compare the average description length across the categories.",
        "Which category has the highest percentage of books marked as Out of stock?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag_qa.answer_query(query)
        print("Method:", result.get('method', 'N/A'))
        print("Answer:", result.get('answer', 'N/A'))
        print("Justification:", result.get('justification', 'N/A'))

if __name__ == "__main__":
    main()