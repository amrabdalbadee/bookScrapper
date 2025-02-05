# Project Setup Guide

## Prerequisites
- Python 3.8+
- pip or conda
- Git

## Installation Steps

1. Clone the Repository
```bash
git clone https://github.com/yourusername/book-data-analysis.git
cd book-data-analysis
```

2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Environment Configuration
- Create a `.env` file in the project root
- Add necessary API keys in src/pipeline/run_qa.py
```
GEMINI_API_KEY=your_gemini_api_key
```