# Usage Guide

## Web Scraping
```bash
# Scrape books from specific categories
python src/pipeline/run_scraper.py --categories Travel Mystery
```

## Question Answering
```bash
# Run QA pipeline with different methods
python src/pipeline/run_qa.py --query "What are mystery books under $20?" --method rule_based
python src/pipeline/run_qa.py --query "Best historical fiction books" --method rag
#Batch Queries
python src/pipeline/run_qa.py --method unsupervised --query-file queries.txt

```

## Example Queries
- Categorical: "Are there travel books available?"
- Numerical: "How many mystery books cost less than $15?"
- Comparative: "Which category has the most 5-star rated books?"

## Supported QA Methods
1. Rule-Based QA
2. LLM Fine-Tuning
3. Retrieval Augmented Generation (RAG)
4. Gemini AI Integration

## Using Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Hugging Face Spaces
Visit: `https://huggingface.co/spaces/yourusername/book-analysis`