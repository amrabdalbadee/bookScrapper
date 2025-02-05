# API Reference

## BookSpider
### `BookSpider`
- **Purpose**: Web scraper for book data
- **Methods**:
  - `parse(response)`: Parse homepage and category links
  - `parse_category(response)`: Extract book listings
  - `parse_book(response)`: Extract individual book details

## BookDataHandler
### `save_as_csv(data, filepath)`
- **Parameters**:
  - `data`: List of book dictionaries
  - `filepath`: Output CSV path
- **Returns**: None

### `save_as_json(data, filepath)`
- **Parameters**:
  - `data`: List of book dictionaries
  - `filepath`: Output JSON path
- **Returns**: None

### `save_as_pickle(data, filepath)`
- **Parameters**:
  - `data`: List of book dictionaries
  - `filepath`: Output Pickle path
- **Returns**: None

## BookDataProcessor
### `process_availability(availability_text)`
- **Parameters**:
  - `availability_text`: Raw availability string
- **Returns**: 
  - `dict`: Processed availability information

### `process_rating(rating_text)`
- **Parameters**:
  - `rating_text`: Raw rating string
- **Returns**: 
  - `int`: Numeric rating (1-5)

## QAPipeline
### `answer_query(query, method='rule_based')`
- **Parameters**:
  - `query`: Natural language query
  - `method`: QA method to use
- **Returns**:
  - `dict`: Query results with answer and method used