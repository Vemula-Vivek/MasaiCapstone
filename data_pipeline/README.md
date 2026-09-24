# Module 1: Data Pipeline

## Run

From the repository root, create and activate a virtual environment, then run:

```bash
python3 -m pip install -r requirements.txt
python3 data_pipeline/data_pipeline.py
```

## Data scope

The script uses `requests` and Beautiful Soup to scrape every listing page in
the `Classics`, `Philosophy`, and `Fiction` categories at Books to Scrape. It
therefore produces more than the required 60 records without manual copying.

## Cleaning decisions

- `price_gbp` is parsed to `float` after removing the GBP symbol.
- `star_rating` values (`One` through `Five`) are converted to integer `rating`
  values from 1 through 5.
- `availability` is parsed to Boolean `in_stock`. SQLite stores that Boolean as
  `1` for true and `0` for false.
- Rows with an invalid price, rating, or availability are dropped and recorded
  in `query_results/dropped_rows.csv`. This avoids inventing catalog prices or
  ratings through imputation.
- The project-defined fixed conversion is **1 GBP = 105.50 INR**. It is an
  assignment baseline, not a live exchange rate.

## Database and outputs

`books.db` is recreated as a current scrape snapshot every time the script
runs. It has normalized `categories` and `books` tables linked through
`category_id`.

The script saves raw and cleaned data, five SQL query strings, their CSV
outputs, and the SQL JOIN versus pandas `merge` comparison in
`data_pipeline/query_results/`.
