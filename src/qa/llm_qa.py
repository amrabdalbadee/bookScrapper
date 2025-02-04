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
    TrainingArguments
)
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from src.processing.data_handler import BookDataHandler

class LocalLLMQuestionAnswering:
    def __init__(
        self, 
        books_data: str = "data/books_data_20250202_163005.csv", 
        unsupervised_model: str = "distilgpt2",
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
        Generate structured QA pairs for supervised fine-tuning.
        
        Returns:
            List[Dict[str, str]]: Question-answer pairs with justifications
        """
        qa_pairs = []
        
        # Categorical questions
        for category in self.data['category'].unique():
            # Out of stock question
            out_of_stock = self.data[
                (self.data['category'] == category) & 
                (self.data['is_available'] == 0)
            ]
            qa_pairs.append({
                "question": f"Are there any books in the {category} category that are marked as Out of stock?",
                "answer": "Yes" if len(out_of_stock) > 0 else "No",
                "justification": (
                    f"{len(out_of_stock)} out of {len(self.data[self.data['category'] == category])} "
                    f"books in {category} category are out of stock."
                )
            })
            
            # Price-related question
            category_books = self.data[self.data['category'] == category]
            avg_price = category_books['price'].mean()
            qa_pairs.append({
                "question": f"What is the average price of books in the {category} category?",
                "answer": f"£{avg_price:.2f}",
                "justification": f"Calculated from {len(category_books)} books in {category} category."
            })
        
        return qa_pairs
    
    def train_unsupervised(
        self, 
        num_epochs: int = 3, 
        save_path: str = "./unsupervised_model"
    ):
        """
        Train model in unsupervised manner using language modeling.
        
        Args:
            num_epochs (int): Number of training epochs
            save_path (str): Directory to save trained model
        
        Returns:
            Tuple of trained model and tokenizer
        """
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
            output_dir=save_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
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
        save_path: str = "./supervised_model"
    ):
        """
        Train model in supervised manner using QA pairs.
        
        Args:
            num_epochs (int): Number of training epochs
            save_path (str): Directory to save trained model
        
        Returns:
            Tuple of trained model and tokenizer
        """
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
            output_dir=save_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir='./logs',
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
                temperature=0.7
            )
        
        # Decode and return
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main execution
def main():
    books_data = "data/books_data_20250202_163005.csv"
    
    # Initialize QA system
    qa_system = LocalLLMQuestionAnswering(books_data)
    
    # Train unsupervised model
    print("Training Unsupervised Model...")
    unsup_model, unsup_tokenizer = qa_system.train_unsupervised(num_epochs=2)
    
    # Train supervised model
    print("\nTraining Supervised Model...")
    sup_model, sup_tokenizer = qa_system.train_supervised(num_epochs=2)
    
    # Test questions
    test_questions = [
        "Are there any books in the Travel category that are marked as Out of stock?",
        "What is the average price of books in the Mystery category?"
    ]
    
    print("\n--- Unsupervised Model Answers ---")
    for q in test_questions:
        answer = qa_system.generate_answer(q, unsup_model, unsup_tokenizer)
        print(f"Q: {q}")
        print(f"A: {answer}\n")
    
    print("\n--- Supervised Model Answers ---")
    for q in test_questions:
        answer = qa_system.generate_answer(q, sup_model, sup_tokenizer)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    main()