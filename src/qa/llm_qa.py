import os
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments,
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from src.processing.data_handler import BookDataHandler
import warnings
import re

warnings.simplefilter("ignore")

class LocalLLMQuestionAnswering:
    def __init__(
        self, 
        books_data: str = "data/books_data_20250202_163005.csv", 
        #unsupervised_model: str = "distilgpt2",
        unsupervised_model: str = "microsoft/MiniLM-L12-H384-uncased",
        #supervised_model: str = "google/flan-t5-small",
        #supervised_model: str = "facebook/opt-350m",
        supervised_model: str = "google/flan-t5-small",
        device: Optional[str] = None
    ):
        """
        Initialize QA system with local LLM fine-tuning capabilities.
        
        Args:
            books_data (List[Dict]): Raw book data for training
            unsupervised_model (str): Base model for unsupervised learning
            supervised_model (str): Base model for supervised learning
            device (str, optional): Compute device (auto-detected if None)
        """
        # Set up data
        self.data_handler = BookDataHandler()
        self.data = self.data_handler.load_csv(books_data)
        
        # Device configuration
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Models and tokenizers
        self.unsupervised_model_name = unsupervised_model
        self.supervised_model_name = supervised_model
        
        # Semantic search transformer
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.unsupervised_checkpoint = "./unsupervised_model_checkpoint"
        self.supervised_checkpoint = "./supervised_model_checkpoint"

    
    def _clean_generated_text(self, text: str) -> str:
        """
        Clean and process generated text to extract meaningful answers.
        
        Args:
            text (str): Raw generated text
        
        Returns:
            str: Cleaned and processed answer
        """
        # Remove the original question if it's part of the generated text
        text = text.replace(f"Question: ", "").strip()
        
        # Extract specific patterns for answers
        patterns = [
            r"Answer:\s*(.+?)(?:\.|$)",  # Extract text after "Answer:"
            r"(\w+(?:\s+\w+)*)\.",  # Extract first meaningful phrase
            r"^(.+?)(?:\.|$)"  # Fallback to first meaningful text
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                cleaned_text = match.group(1).strip()
                if cleaned_text and len(cleaned_text) > 1:
                    return cleaned_text
        
        return text.strip()
    

    def _prepare_unsupervised_texts(self) -> List[str]:
        """
        Prepare texts for unsupervised learning using natural language templates.
        
        Returns:
            List[str]: Processed text descriptions
        """
        texts = []
        
        for _, row in self.data.iterrows():
            # Book description template
            book_desc = (
                f"Book '{row['title']}' in the {row['category']} category "
                f"costs £{row['price']:.2f}. "
                f"It has a rating of {row['star_rating']} stars. "
                f"Currently, there are {row['stock_count']} in stock. "
                f"It is {'in' if row['is_available'] else 'out of'} stock. "
                f"Description: {row['description']}"
            )
            texts.append(book_desc)
            
            # Category statistics template
            category_data = self.data[self.data['category'] == row['category']]
            cat_stats = (
                f"In the {row['category']} category: "
                f"Average price is £{category_data['price'].mean():.2f}. "
                f"Average rating is {category_data['star_rating'].mean():.1f} stars. "
                f"Total books: {len(category_data)}. "
                f"Out of stock books: {(category_data['is_available'] == 0).sum()}. "
                f"Total available stock: {category_data['stock_count'].sum()}."
            )
            texts.append(cat_stats)
        
        return texts
    
    def _generate_supervised_qa_pairs(self) -> List[Dict[str, str]]:
        """
        Generate comprehensive QA pairs for supervised fine-tuning.
        """
        qa_pairs = []
        
        # Categorical Questions
        for category in self.data['category'].unique():
            # Out of stock questions
            out_of_stock = self.data[
                (self.data['category'] == category) & 
                (self.data['is_available'] == 0)
            ]
            qa_pairs.append({
                "question": f"Are there any books in the {category} category that are marked as Out of stock?",
                "answer": "Yes" if len(out_of_stock) > 0 else "No",
                "justification": f"{len(out_of_stock)} out of {len(self.data[self.data['category'] == category])} books in {category} are out of stock."
            })
            
            # 5-star rating question
            five_star_books = self.data[
                (self.data['category'] == category) & 
                (self.data['star_rating'] == 5)
            ]
            qa_pairs.append({
                "question": f"Does the {category} category contain books with a 5-star rating?",
                "answer": "Yes" if len(five_star_books) > 0 else "No",
                "justification": f"{len(five_star_books)} 5-star rated books found in {category}."
            })
            
            # Books below £10
            low_price_books = self.data[
                (self.data['category'] == category) & 
                (self.data['price'] < 10)
            ]
            qa_pairs.append({
                "question": f"Are there books in the {category} category priced below £10?",
                "answer": "Yes" if len(low_price_books) > 0 else "No",
                "justification": f"{len(low_price_books)} books priced below £10 in {category}."
            })
            
            # Numerical Data Questions
            category_books = self.data[self.data['category'] == category]
            avg_price = category_books['price'].mean()
            min_price = category_books['price'].min()
            max_price = category_books['price'].max()
            total_stock = category_books['stock_count'].sum()
            total_value = category_books['price'].sum()
            
            qa_pairs.extend([
                {
                    "question": f"What is the average price of books in the {category} category?",
                    "answer": f"£{avg_price:.2f}",
                    "justification": f"Calculated from {len(category_books)} books in {category}."
                },
                {
                    "question": f"What is the price range for books in the {category} category?",
                    "answer": f"£{min_price:.2f} to £{max_price:.2f}",
                    "justification": f"Minimum and maximum prices from {len(category_books)} books in {category}."
                },
                {
                    "question": f"How many books are available in stock in the {category} category?",
                    "answer": str(total_stock),
                    "justification": f"Total stock count for {len(category_books)} books in {category}."
                },
                {
                    "question": f"What is the total value of books in the {category} category?",
                    "answer": f"£{total_value:.2f}",
                    "justification": f"Sum of prices for {len(category_books)} books in {category}."
                }
            ])
        
        # Hybrid and Comparative Questions
        # Average Price Comparison
        category_avg_prices = self.data.groupby('category')['price'].mean()
        highest_price_category = category_avg_prices.idxmax()
        qa_pairs.append({
            "question": "Which category has the highest average price of books?",
            "answer": highest_price_category,
            "justification": f"{highest_price_category} has the highest average price of £{category_avg_prices.max():.2f}."
        })
        
        # Categories with high-priced books
        high_price_categories = []
        for category in self.data['category'].unique():
            category_books = self.data[self.data['category'] == category]
            high_price_ratio = (category_books['price'] > 30).mean()
            if high_price_ratio > 0.5:
                high_price_categories.append(category)
        
        qa_pairs.append({
            "question": "Which categories have more than 50% of their books priced above £30?",
            "answer": str(high_price_categories) if high_price_categories else "None",
            "justification": f"Categories with over 50% books above £30: {high_price_categories}."
        })
        
        # Out of stock percentage comparison
        out_of_stock_percentages = {}
        for category in self.data['category'].unique():
            category_books = self.data[self.data['category'] == category]
            out_of_stock_percentages[category] = (category_books['is_available'] == 0).mean() * 100
        
        highest_out_of_stock_category = max(out_of_stock_percentages, key=out_of_stock_percentages.get)
        qa_pairs.append({
            "question": "Which category has the highest percentage of books marked as Out of stock?",
            "answer": highest_out_of_stock_category,
            "justification": f"{highest_out_of_stock_category} has {out_of_stock_percentages[highest_out_of_stock_category]:.2f}% out of stock books."
        })

        # Add sample-based questions
        for _, row in self.data.iloc[:10].iterrows():
            # Questions based on book descriptions
            qa_pairs.extend([
                {
                    "question": f"What is the book '{row['title']}' about?",
                    "answer": row['description'],
                    "justification": f"Description from the book in the {row['category']} category"
                },
                {
                    "question": f"What category does the book '{row['title']}' belong to?",
                    "answer": row['category'],
                    "justification": f"Categorized under {row['category']}"
                },
                {
                    "question": f"What is the price of the book '{row['title']}'?",
                    "answer": f"£{row['price']:.2f}",
                    "justification": f"Price for the book in the {row['category']} category"
                },
                {
                    "question": f"Is '{row['title']}' currently in stock?",
                    "answer": "Yes" if row['is_available'] else "No",
                    "justification": f"Stock status for the book in the {row['category']} category"
                }
            ])
        
        return qa_pairs
    
    def _is_model_trained(self, checkpoint_path: str) -> bool:
        """Check if model is already trained."""
        return os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin"))
    
    def train_unsupervised(
        self, 
        num_epochs: int = 3, 
        save_path: str = "./unsupervised_model",
        force_retrain: bool = False
    ):
        """
        Train model in unsupervised manner using language modeling.
        
        Args:
            num_epochs (int): Number of training epochs
            save_path (str): Directory to save trained model
        
        Returns:
            Tuple of trained model and tokenizer
        """
        # Skip training if already completed
        if not force_retrain and self._is_model_trained(self.unsupervised_checkpoint):
            print("Loading pre-trained unsupervised model...")
            return AutoModelForSequenceClassification.from_pretrained(self.unsupervised_checkpoint)
        
        # Prepare texts
        texts = self._prepare_unsupervised_texts()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.unsupervised_model_name)
        model = AutoModelForCausalLM.from_pretrained(self.unsupervised_model_name)
        
        # Tokenization setup
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=256, 
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.unsupervised_checkpoint,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=1,
            learning_rate=5e-5
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        return model, tokenizer
    
    def train_supervised(
        self, 
        num_epochs: int = 3, 
        save_path: str = "./supervised_model",
        force_retrain: bool = False
    ):
        """
        Train model in supervised manner using QA pairs.
        
        Args:
            num_epochs (int): Number of training epochs
            save_path (str): Directory to save trained model
        
        Returns:
            Tuple of trained model and tokenizer
        """
        if not force_retrain and self._is_model_trained(self.supervised_checkpoint):
            print("Loading pre-trained supervised model...")
            return AutoModelForSeq2SeqLM.from_pretrained(self.supervised_checkpoint)
        
        # Generate QA pairs
        qa_pairs = self._generate_supervised_qa_pairs()
        
        # Create dataset
        dataset = Dataset.from_dict({
            "question": [qa['question'] for qa in qa_pairs],
            "answer": [f"Answer: {qa['answer']}. Justification: {qa['justification']}" for qa in qa_pairs]
        })
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.supervised_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.supervised_model_name)
        
        # Tokenization function
        def tokenize_function(examples):
            # Tokenize inputs (questions)
            inputs = tokenizer(
                examples["question"], 
                max_length=128, 
                truncation=True, 
                padding="max_length"
            )
            
            # Tokenize outputs (answers with justification)
            labels = tokenizer(
                examples["answer"], 
                max_length=256, 
                truncation=True, 
                padding="max_length"
            )
            
            inputs["labels"] = labels["input_ids"]
            return inputs
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.supervised_checkpoint,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=1,
            logging_dir='./logs',
            learning_rate=5e-5
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        return model, tokenizer
    
    def generate_answer(
        self, 
        question: str, 
        model, 
        tokenizer, 
        max_length: int = 256
    ) -> str:
        """
        Generate answer for a given question using trained model.
        
        Args:
            question (str): Input question
            model: Trained LLM model
            tokenizer: Model's tokenizer
            max_length (int): Maximum generation length
        
        Returns:
            str: Generated answer with justification
        """
        try:
            # Prepare input
            inputs = tokenizer(
                f"Question: {question}", 
                return_tensors="pt", 
                max_length=128, 
                truncation=True
            ).to(self.device)
            
            model = model.to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
            
            # Decode and clean the generated text
            raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_generated_text(raw_text)
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Unable to generate an answer."
        
# Main execution
def main():
    books_data = "data/books_data_20250202_163005.csv"
    
    # Initialize QA system
    qa_system = LocalLLMQuestionAnswering(books_data)
    
    # # Train unsupervised model
    # print("Training Unsupervised Model...")
    # unsup_model, unsup_tokenizer = qa_system.train_unsupervised(num_epochs=2)
    
    # Train supervised model
    print("\nTraining Supervised Model...")
    sup_model, sup_tokenizer = qa_system.train_supervised(num_epochs=5)
    
    # Test questions
    test_questions = [
        # Categorical Questions
        "Are there any books in the Travel category that are marked as Out of stock?",
        "Does the Mystery category contain books with a 5-star rating?",
        "Are there books in the Classics category priced below £10?",
        "Are more than 50% of books in the Mystery category priced above £20?",
        
        # Numerical Data Questions
        "What is the average price of books across each category?",
        "What is the price range for books in the Historical Fiction category?",
        "How many books are available in stock across the categories?",
        "What is the total value of books in the Travel category?",
        
        # Hybrid Questions
        "Which category has the highest average price of books?",
        "Which categories have more than 50% of their books priced above £30?",
        "Which category has the highest percentage of books marked as Out of stock?"
    ]
    
    # print("\n--- Unsupervised Model Answers ---")
    # for q in test_questions:
    #     answer = qa_system.generate_answer(q, unsup_model, unsup_tokenizer)
    #     print(f"Q: {q}")
    #     print(f"A: {answer}")
    
    print("\n--- Supervised Model Answers ---")
    for q in test_questions:
        answer = qa_system.generate_answer(q, sup_model, sup_tokenizer)
        print(f"Q: {q}")
        print(f"A: {answer}")

if __name__ == "__main__":
    main()