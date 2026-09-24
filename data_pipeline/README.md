# Data Pipeline

This module collects book data from books.toscrape.com, cleans it, converts GBP to INR at 1 GBP = 105.50 INR, and stores the result in SQLite. It also runs a few SQL queries and compares the SQL join with a pandas merge.

## Setup

From the repo root:

```bash
pip install -r requirements.txt
python data_pipeline/pipeline.py
```

## What the script does

- The scraper visits four categories and collects more than 60 rows.
- Price parsing strips the currency symbol and converts the value to a float.
- Star ratings are mapped from text labels such as "Three" to integer values 1–5.
- Availability is parsed into a boolean in_stock column.
- Rows with malformed scraped values are dropped instead of being filled with guesses.
- The conversion rate is fixed at 1 GBP = 105.50 INR.

## Output

The script creates a SQLite database at `data_pipeline/zepto_books.db` and a file at `data_pipeline/queries_output.txt` containing the SQL queries and their outputs.

## SQL coverage

The generated SQL covers SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, and a JOIN between `categories` and `books`. The BETWEEN example uses INR 1000-2000 because the converted prices start above INR 1000. The script stops with a clear error if a required query returns no rows.
