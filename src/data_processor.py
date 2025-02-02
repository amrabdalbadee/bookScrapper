# src/data_processor.py
import pandas as pd
import re
from typing import Union, List, Dict

class BookDataProcessor:
    """
    Process and clean book data from the scraper output.
    """
    
    @staticmethod
    def process_availability(availability: Union[List[str], str]) -> tuple:
        """
        Process availability string to extract status and count.
        Returns (is_available, count) tuple.
        """
        # Handle if availability is a list
        if isinstance(availability, list):
            # Find the first non-empty string that contains "stock" information
            availability_text = next((text for text in availability if 'stock' in text.lower()), '')
        else:
            availability_text = str(availability)
            
        # Check if in stock
        is_available = 1 if 'In stock' in availability_text else 0
        
        # Extract count if available
        count_match = re.search(r'(\d+) available', availability_text)
        count = int(count_match.group(1)) if count_match else 0 if not is_available else 1
        
        return is_available, count

    @staticmethod
    def process_rating(rating: str) -> int:
        """
        Convert star rating text to integer value (1-5).
        """
        if not rating or not isinstance(rating, str):
            return 0
            
        rating_map = {
            'One': 1,
            'Two': 2,
            'Three': 3,
            'Four': 4,
            'Five': 5
        }
        
        # Extract the rating text
        rating_text = rating.replace('star-rating ', '')
        return rating_map.get(rating_text, 0)

    @staticmethod
    def process_price(price: str) -> float:
        """
        Convert price string to float value.
        """
        if not price or not isinstance(price, str):
            return 0.0
            
        # Remove currency symbol and convert to float
        return float(price.replace('£', '').strip())

    @staticmethod
    def clean_description(description: str) -> str:
        """
        Clean book description text.
        """
        if not description:
            return ""
            
        # Remove '...more' and extra whitespace
        cleaned = re.sub(r'\.\.\.more$', '', description)
        return cleaned.strip()

    def process_book_data(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Process book data and return cleaned DataFrame.
        """
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Process availability
        availability_processed = df['availability'].apply(self.process_availability)
        df['is_available'] = availability_processed.apply(lambda x: x[0])
        df['stock_count'] = availability_processed.apply(lambda x: x[1])
        df.drop('availability', axis=1, inplace=True)
        
        # Process rating
        df['star_rating'] = df['rating'].apply(self.process_rating)
        df.drop('rating', axis=1, inplace=True)
        
        # Process price
        df['price'] = df['price'].apply(self.process_price)
        
        # Clean description
        df['description'] = df['description'].apply(self.clean_description)
        
        return df

    def process_dataset(self, filepath: str, output_filepath: str = None) -> pd.DataFrame:
        """
        Process entire dataset from file and optionally save results.
        """
        # Load data based on file extension
        if filepath.endswith('.json'):
            df = pd.read_json(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format")
            
        # Process the data
        processed_df = self.process_book_data(df)
        
        # Save if output path provided
        if output_filepath:
            if output_filepath.endswith('.json'):
                processed_df.to_json(output_filepath, orient='records', indent=2)
            elif output_filepath.endswith('.csv'):
                processed_df.to_csv(output_filepath, index=False)
                
        return processed_df