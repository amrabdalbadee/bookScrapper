import streamlit as st
import time
import os
import sys
from typing import Dict, Any

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.run_qa import QAPipeline

def load_suggested_queries() -> Dict[str, list]:
    """
    Predefined set of suggested queries for different question types.
    
    Returns:
        Dict of query categories and their example queries
    """
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

def setup_streamlit_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Book QA Intelligence",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        border: 2px solid #3498db;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Setup Streamlit page
    setup_streamlit_page()
    
    # Header and Introduction
    st.title("📚 Book QA Intelligence")
    st.markdown("""
    ### Intelligent Book Dataset Query System
    
    Explore and extract insights from our comprehensive book dataset using 
    advanced AI-powered question answering techniques. Choose from multiple 
    intelligent methods to get precise answers to your queries.
    """)
    
    # Sidebar for Method Selection
    st.sidebar.header("Query Settings")
    qa_method = st.sidebar.selectbox(
        "Select QA Method",
        ["Rule-Based", "Unsupervised LLM", "Supervised LLM", "RAG", "GEMINI"],
        index=0
    )
    
    # Suggested Queries Section
    st.sidebar.header("Suggested Queries")
    suggested_queries = load_suggested_queries()
    
    for category, queries in suggested_queries.items():
        with st.sidebar.expander(category):
            for query in queries:
                if st.button(query, key=query):
                    st.session_state.user_query = query
    
    # Query Input
    st.header("Ask a Question")
    user_query = st.text_input(
        "Enter your query about the book dataset", 
        key="user_query",
        value=st.session_state.get("user_query", "")
    )
    
    # Submit Button
    submit_button = st.button("Submit Query")
    
    # Initialize QA Pipeline only when the method and query are selected
    if user_query and submit_button:
        method_map = {
            "Rule-Based": "rule_based",
            "Unsupervised LLM": "unsupervised", 
            "Supervised LLM": "supervised",
            "RAG": "rag",
            "GEMINI": "gemini"
        }
        
        pipeline = QAPipeline(method=method_map[qa_method])
        
        # Progress Indicators
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
            status_text.text("Processing query...")
            result = pipeline.answer_query(user_query)
            
            # Compute time taken
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)
            
            # Display Results
            st.success("Query Processed Successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Method Used", result['method'].capitalize())
            with col2:
                st.metric("Time Taken", f"{time_taken} seconds")
            
            st.subheader("Answer")
            st.info(result.get('answer', 'No answer could be generated.'))
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    elif not submit_button:
        st.warning("Please click the 'Submit Query' button to get the answer.")
    else:
        st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
