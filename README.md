# Book Data Analysis System

A comprehensive system for scraping, processing, and analyzing book data using advanced NLP techniques and multiple question-answering approaches.

## Features

- **Web Scraping**: Scalable scraping from books.toscrape.com
- **Data Processing**: Multi-format data handling (CSV, JSON, Pickle)
- **Question Answering**:
  - Rule-based analysis
  - LLM fine-tuning
  - Retrieval Augmented Generation (RAG)
  - Google Gemini AI integration
- **User Interfaces**: Streamlit app and Hugging Face Spaces deployment

## Installation

```bash
# Clone repository
git clone https://github.com/amrabdalbadee/bookScrapper
cd book-analysis-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## Quick Start

```bash
# Run web scraper
python src/pipeline/run_scraper.py

# Process queries
python src/pipeline/run_qa.py --query "What are mystery books under $20?" --method rule_based

# Launch Streamlit app
streamlit run app/app.py
```

## Project Structure

```
├── src/
│   ├── spider/        # Web scraping components
│   ├── processing/    # Data processing modules
│   ├── qa/           # Question answering systems
│   └── pipeline/     # Integration pipelines
├── app/
│   ├── app.py #streamlit app
│   └── hf_app.py #huggingface spaces app
├── data/             # Data storage
└── docs/             # Documentation
```

## Documentation

- [Overview](docs/overview.md)
- [Setup Guide](docs/setup.md)
- [API Reference](docs/api_reference.md)
- [Usage Guide](docs/usage.md)
- [Design Decisions](docs/design_decisions.md)
- [Deployment Guide](docs/deployment.md)
- [Challenges](docs/challenges.md)

## Models Used

- Question Answering:
  - Rule-based: CodeLlama-7b-Python
  - Supervised: Flan-T5-Small
  - Embeddings: MiniLM-L12-H384

## Requirements

- Python 3.8+
- 8GB RAM minimum
- CUDA-compatible GPU (optional)
- Google Gemini API key 

## License

MIT

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## Contact

- GitHub: [@amrabdalbadee](https://github.com/amrabdalbadee)
- Email: amrabdalbadee@gmail.com