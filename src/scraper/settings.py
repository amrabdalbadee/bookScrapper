# src/scraper/settings.py
BOT_NAME = 'bookscraper'
SPIDER_MODULES = ['src.scraper']
NEWSPIDER_MODULE = 'src.scraper'
USER_AGENT = 'Mozilla/5.0 (compatible; BookAnalyzer/1.0)'
ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = 16
DOWNLOAD_DELAY = 1
COOKIES_ENABLED = False