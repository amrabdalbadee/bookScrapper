# src/run_scraper.py
import asyncio
import logging
from pipeline.pipeline import BookScrapingPipeline
from twisted.internet import reactor, defer

def run_spider():
    """Run the spider and return a deferred"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create pipeline instance
        pipeline = BookScrapingPipeline()
        
        # Run with specific formats and categories
        d = pipeline.scrape_and_save(
            export_formats=['csv', 'json'],
            categories=['Travel', 'Mystery', 'Historical Fiction', 'Classics']
        )
        
        # Add callbacks
        d.addCallback(lambda result: logger.info(f"Files saved: {result}"))
        d.addErrback(lambda failure: logger.error(f"Error: {failure}"))
        
        return d
        
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the spider
    defer.ensureDeferred(run_spider())
    # Start the reactor
    reactor.run()