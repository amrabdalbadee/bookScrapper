# src/pipeline.py
import logging
from datetime import datetime
from pathlib import Path
from scrapy.crawler import CrawlerRunner
from twisted.internet import defer
from scrapy.utils.project import get_project_settings
import json

from scraper.spider import BookSpider
from data_handler import BookDataHandler
from data_processor import BookDataProcessor


class BookScrapingPipeline:
    """
    A pipeline class that combines scraping and data handling operations.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_handler = BookDataHandler(data_dir)
        self.scraped_data = []
        self.processed_data = []
        self._setup_logging()
        self._create_directories()
        
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        Path('logs').mkdir(exist_ok=True)
        Path('data').mkdir(exist_ok=True)
        
    def _setup_crawler(self) -> CrawlerRunner:
        """Set up the Scrapy crawler with appropriate settings."""
        settings = get_project_settings()
        settings.update({
            'FEEDS': {
                'data/temp_scrape.json': {
                    'format': 'json',
                    'overwrite': True
                }
            },
            'LOG_LEVEL': 'INFO',
            'ROBOTSTXT_OBEY': True,
            'CONCURRENT_REQUESTS': 16,
            'DOWNLOAD_DELAY': 1,
            'COOKIES_ENABLED': False
        })
        return CrawlerRunner(settings)

    #Uses Scrapy's async Twisted engine
    @defer.inlineCallbacks
    def scrape_and_save(self, export_formats=None, categories=None):
        """
        Main method to run the scraping process and save data.
        Returns a deferred that will fire when the scraping is complete.
        """
        export_formats = export_formats or ['json']
        self.logger.info("Starting scraping process...")
        
        try:
            # Configure spider settings
            if categories:
                BookSpider.target_categories = categories
            
            # Set up and run the crawler
            runner = self._setup_crawler()
            yield runner.crawl(BookSpider)
            
            # Process the scraped data
            saved_files = yield self._process_scraped_data(export_formats)
            
            defer.returnValue(saved_files)
            
        except Exception as e:
            self.logger.error(f"Error in scraping pipeline: {str(e)}")
            raise

    @defer.inlineCallbacks
    def _process_scraped_data(self, export_formats):
        """Process and save the scraped data in specified formats."""
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Wait a moment to ensure the file is written
            yield defer.succeed(None)
            
            # Load the temporary scraped data
            with open('data/temp_scrape.json', 'r') as f:
                self.scraped_data = json.load(f)
           
            # Process the data
            processor = BookDataProcessor()
            processed_data = processor.process_book_data(self.scraped_data)
            print(processed_data.columns)      
            self.processed_data = self.data_handler.df_to_json(processed_data)
            
            # Save in each requested format
            for format_type in export_formats:
                if format_type.lower() == 'csv':
                    filepath = self.data_handler.save_as_csv(
                        self.processed_data, 
                        f'books_data_{timestamp}.csv'
                    )
                    saved_files['csv'] = filepath
                    
                elif format_type.lower() == 'json':
                    filepath = self.data_handler.save_as_json(
                        self.processed_data, 
                        f'books_data_{timestamp}.json'
                    )
                    saved_files['json'] = filepath
                    
                elif format_type.lower() == 'pickle':
                    filepath = self.data_handler.save_as_pickle(
                        self.processed_data, 
                        f'books_data_{timestamp}.pkl'
                    )
                    saved_files['pickle'] = filepath
                    
                elif format_type.lower() == 'dataframe':
                    saved_files['dataframe'] = self.data_handler.to_dataframe(self.processed_data)
            
            # Clean up temporary file
            Path('data/temp_scrape.json').unlink(missing_ok=True)
            
            self.logger.info(f"Data saved successfully in formats: {', '.join(export_formats)}")
            defer.returnValue(saved_files)
            
        except Exception as e:
            self.logger.error(f"Error processing scraped data: {str(e)}")
            raise