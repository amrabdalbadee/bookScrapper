import streamlit as st
import time
import os
import sys
from typing import Dict, Any

# Hugging Face specific imports
from huggingface_hub import HfApi, HfFolder

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.run_qa import QAPipeline

def load_suggested_queries() -> Dict[str, list]:
    """Predefined set of suggested queries for different question types."""
    return {
        "Categorical Questions": [
            "Are there any books in the 'Travel' category that are marked as Out of stock?",
            "Does the 'Mystery' category contain books with a 5-star rating?",
            "Are there books in the 'Classics' category priced below £10?",
            "Are more than 50% of books in the 'Mystery' category priced above £20?"
        ],
        "Numerical Data Extraction": [
            "What is the average price of books across each category?",
            "What is the price range for books in the 'Historical Fiction' category?",
            "How many books are available in stock across the categories?",
            "What is the total value of all books in the 'Travel' category?"
        ],
        "Hybrid Questions": [
            "Which category has the highest average price of books?",
            "Which categories have more than 50% of their books priced above £30?",
            "Compare the average description length across the categories.",
            "Which category has the highest percentage of books marked as Out of stock?"
        ]
    }

def setup_huggingface_page():
    """Configure Hugging Face Spaces Streamlit page settings."""
    st.set_page_config(
        page_title="Book QA Intelligence",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Hugging Face specific styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f4f4f4;
        font-family: 'Hugging Face', sans-serif;
    }
    .stTextInput > div > div > input {
        border: 2px solid #FF6D37;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #FF6D37;
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Setup Hugging Face page
    setup_huggingface_page()
    
    # Hugging Face Spaces Header
    st.title("🤗 Book QA Intelligence on Hugging Face")
    st.markdown("""
    ### AI-Powered Book Dataset Query System
    Explore book insights using advanced question-answering techniques.
    """)
    
    # Sidebar for Method Selection
    st.sidebar.header("Query Configuration")
    qa_method = st.sidebar.selectbox(
        "Select QA Method",
        ["Rule-Based", "Unsupervised LLM", "Supervised LLM", "RAG"],
        index=0
    )
    
    # Initialize QA Pipeline
    method_map = {
        "Rule-Based": "rule_based",
        "Unsupervised LLM": "unsupervised", 
        "Supervised LLM": "supervised",
        "RAG": "rag"
    }
    pipeline = QAPipeline(method=method_map[qa_method])
    
    # Suggested Queries Section
    st.sidebar.header("Quick Query Templates")
    suggested_queries = load_suggested_queries()
    
    for category, queries in suggested_queries.items():
        with st.sidebar.expander(category):
            for query in queries:
                if st.button(query, key=query):
                    st.session_state.user_query = query
    
    # Query Input
    st.header("Book Dataset Query")
    user_query = st.text_input(
        "Enter your book-related query", 
        key="user_query",
        value=st.session_state.get("user_query", "")
    )
    
    # Query Processing
    if st.button("Analyze Query"):
        if user_query:
            # Progress and Status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Time the query
                start_time = time.time()
                
                # Simulate progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                # Process Query
                status_text.text("Analyzing query...")
                result = pipeline.answer_query(user_query)
                
                # Compute time taken
                end_time = time.time()
                time_taken = round(end_time - start_time, 2)
                
                # Display Results
                st.success("Query Processed Successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Analysis Method", result['method'].capitalize())
                with col2:
                    st.metric("Processing Time", f"{time_taken} seconds")
                
                st.subheader("Insights")
                st.info(result.get('answer', 'No insights generated.'))
                
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
        else:
            st.warning("Please provide a query about the book dataset.")

if __name__ == "__main__":
    main()

# Optional: Hugging Face Space Deployment Configuration
def get_space_config():
    """Generate Hugging Face Spaces configuration."""
    return {
        "model_name": "book-qa-intelligence",
        "hardware": "cpu-basic",
        "dependencies": [
            "streamlit",
            "torch",
            "transformers",
            "sentence-transformers",
            "pandas"
        ],
        "license": "Apache-2.0"
    }