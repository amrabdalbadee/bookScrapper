# src/data_handler.py
from typing import Dict, List, Union
import pandas as pd
import json
import csv
from pathlib import Path
import pickle
from datetime import datetime
import logging
import os
from typing import List, Dict, Any


class BookDataHandler:
    """
    A class to handle the storage and loading of scraped book data in various formats.
    Supports CSV, JSON, Pickle, and Pandas DataFrame formats.
    """

    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data handler with a data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_data(self, data: List[Dict]) -> bool:
        """
        Validate the structure of input data.
        Time Complexity: O(n) where n is the number of books
        Space Complexity: O(1)
        """
        #required_fields = {'title', 'category', 'price', 'availability', 'star-rating', 'description'}
        required_fields = {'title', 'category', 'price', 'description', 'is_available', 'stock_count', 'star_rating'}
        
        if not data or not isinstance(data, list):
            return False
            
        for book in data:
            if not all(field in book for field in required_fields):
                return False
        return True
    
    def save_as_csv(self, data: List[Dict], filename: str = None) -> str:
        """
        Save data as CSV file.
        Time Complexity: O(n) where n is the number of books
        Space Complexity: O(n)

        Advantages:

        Human-readable
        Universal compatibility
        Good for large datasets that need to be processed in chunks


        Disadvantages:

        Slower to read/write compared to binary formats
        No native support for complex data types

        Use CSV when:

        You need human-readable data
        You want universal compatibility
        You need to process data in chunks

        """
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
            
        filename = filename or f"books_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.data_dir / filename
        
        try:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Data saved successfully to {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
            raise
    
    def save_as_json(self, data: List[Dict], filename: str = None) -> str:
        """
        Save data as JSON file.
        Time Complexity: O(n) where n is the number of books
        Space Complexity: O(n)

        Advantages:

        Human-readable
        Preserves data types
        Native support for nested structures


        Disadvantages:

        Larger file size compared to binary formats
        Slower than binary formats
        
        Use JSON when:

        You need human-readable data with preserved data types
        You have nested data structures
        You need to share data across different systems
        """
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
            
        filename = filename or f"books_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.data_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Data saved successfully to {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}")
            raise
    
    def save_as_pickle(self, data: List[Dict], filename: str = None) -> str:
        """
        Save data as Pickle file.
        Time Complexity: O(n) where n is the number of books
        Space Complexity: O(n)

        Advantages:

        Fastest read/write operations
        Smallest file size
        Preserves Python objects exactly


        Disadvantages:

        Not human-readable
        Python-specific
        Security concerns with untrusted data

        Use Pickle when:

        You need the fastest possible read/write operations
        You're working entirely within Python
        File size is a concern
        """
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
            
        filename = filename or f"books_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = self.data_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Data saved successfully to {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Error saving Pickle: {e}")
            raise
    
    def to_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """
        Convert data to Pandas DataFrame.
        Time Complexity: O(n) where n is the number of books
        Space Complexity: O(n)

        Advantages:

        Efficient for data analysis
        Rich functionality for data manipulation
        Good for memory-efficient operations on large datasets


        Disadvantages:

        Higher memory usage for small datasets
        Slower for simple operations compared to basic Python data structures

        Use DataFrame when:

        You need to perform complex data analysis
        You need to filter, group, or aggregate data
        You need to perform statistical operations
        """
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
            
        return pd.DataFrame(data)
    
    def load_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file.
        Time Complexity: O(n) where n is the number of books
        Space Complexity: O(n)
        """
        try:
            filepath = Path(filepath)
            df = pd.read_csv(filepath)
            self.logger.info(f"Data loaded successfully from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_json(self, filepath: Union[str, Path]) -> List[Dict]:
        """
        Load data from JSON file.
        Time Complexity: O(n) where n is the number of books
        Space Complexity: O(n)
        """
        try:
            filepath = Path(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Data loaded successfully from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading JSON: {e}")
            raise
    
    def load_pickle(self, filepath: Union[str, Path]) -> List[Dict]:
        """
        Load data from Pickle file.
        Time Complexity: O(n) where n is the number of books
        Space Complexity: O(n)
        """
        try:
            filepath = Path(filepath)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.logger.info(f"Data loaded successfully from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading Pickle: {e}")
            raise
    
    def df_to_json(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        json_data = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            json_data.append(row_dict)
        return json_data

    def get_storage_stats(self) -> Dict:
        """
        Get statistics about stored data files.
        Time Complexity: O(m) where m is the number of files
        Space Complexity: O(m)
        """
        stats = {
            'csv_files': [],
            'json_files': [],
            'pickle_files': [],
            'total_size': 0
        }
        
        for file in self.data_dir.glob('*'):
            size = os.path.getsize(file)
            stats['total_size'] += size
            
            if file.suffix == '.csv':
                stats['csv_files'].append({'name': file.name, 'size': size})
            elif file.suffix == '.json':
                stats['json_files'].append({'name': file.name, 'size': size})
            elif file.suffix == '.pkl':
                stats['pickle_files'].append({'name': file.name, 'size': size})
        
        return stats
