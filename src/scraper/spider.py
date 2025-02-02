# src/scraper/spider.py
from scrapy import Spider, Request
from urllib.parse import urljoin

class BookSpider(Spider):
    name = 'bookspider'
    allowed_domains = ['books.toscrape.com']
    start_urls = ['http://books.toscrape.com/']
    
    # Categories we want to scrape
    target_categories = ['Travel', 'Mystery', 'Historical Fiction', 'Classics']
    
    def parse(self, response):
        # Find all category links
        category_links = response.css('div.side_categories ul.nav li ul li a')
        
        for link in category_links:
            category_name = link.css('::text').get().strip()
            if category_name in self.target_categories:
                category_url = urljoin(response.url, link.attrib['href'])
                yield Request(
                    category_url,
                    callback=self.parse_category,
                    meta={'category': category_name}
                )
    
    def parse_category(self, response):
        # Extract books from category page
        books = response.css('article.product_pod')
        
        for book in books:
            book_url = urljoin(response.url, book.css('h3 a::attr(href)').get())
            yield Request(
                book_url,
                callback=self.parse_book,
                meta={'category': response.meta['category']}
            )
        
        # Handle pagination
        next_page = response.css('li.next a::attr(href)').get()
        if next_page:
            next_url = urljoin(response.url, next_page)
            yield Request(
                next_url,
                callback=self.parse_category,
                meta={'category': response.meta['category']}
            )
    
    def parse_book(self, response):
        # Extract all required book details
        yield {
            'title': response.css('div.product_main h1::text').get(),
            'category': response.meta['category'],
            'price': response.css('p.price_color::text').get(),
            'availability': response.css('p.availability::text').getall(),
            'rating': response.css('p.star-rating::attr(class)').get(),
            'description': response.css('div#product_description + p::text').get(),
        }
