"""Scrape, clean, store, and analyse Books to Scrape catalog data."""

import sqlite3
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
TARGET_CATEGORIES = ("Classics", "Philosophy", "Fiction")
GBP_TO_INR = 105.50

BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "books.db"
RESULTS_DIR = BASE_DIR / "query_results"

RATING_MAP = { "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5 }


def get_soup(session, url):
    """Fetch one HTML page and return a parsed BeautifulSoup object."""
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


def scrape_books(session):
    """Scrape every listing page in the three requested categories."""
    home_soup = get_soup(session, BASE_URL)
    category_urls = {
        link.get_text(strip=True): urljoin(BASE_URL, link["href"])
        for link in home_soup.select("div.side_categories ul.nav-list ul a")
        if link.get_text(strip=True) in TARGET_CATEGORIES
    }

    missing_categories = set(TARGET_CATEGORIES) - set(category_urls)
    if missing_categories:
        raise RuntimeError(f"Categories not found: {sorted(missing_categories)}")

    scraped_books = []
    for category_name in TARGET_CATEGORIES:
        page_url = category_urls[category_name]

        # No 60-row cutoff: all pages in each selected category are required.
        while page_url:
            print(f"Scraping {category_name}: {page_url}")
            soup = get_soup(session, page_url)

            for product in soup.select("article.product_pod"):
                title_tag = product.select_one("h3 a")
                price_tag = product.select_one("p.price_color")
                rating_tag = product.select_one("p.star-rating")
                availability_tag = product.select_one("p.availability")

                scraped_books.append(
                    {
                        "category": category_name,
                        "title": title_tag.get("title") if title_tag else None,
                        "price": price_tag.get_text(strip=True) if price_tag else None,
                        # Extract "Three" from class="star-rating Three".
                        "star_rating": (
                            rating_tag.get("class", [])[-1]
                            if rating_tag and rating_tag.get("class")
                            else None
                        ),
                        "availability": (
                            availability_tag.get_text(" ", strip=True)
                            if availability_tag
                            else None
                        ),
                    }
                )

            next_link = soup.select_one("ul.pager li.next a")
            page_url = urljoin(page_url, next_link["href"]) if next_link else None

    return scraped_books


def clean_books(scraped_books):
    """Return typed records and a log of dropped malformed records.

    A row with an invalid price, rating, or availability is dropped rather than
    imputed. Guessing catalog prices or ratings would corrupt the source data.
    """
    cleaned_books = []
    dropped_rows = []

    for book in scraped_books:
        try:
            if not book["title"]:
                raise ValueError("Missing title")
            price_text = book["price"].replace("£", "").replace("Â£", "").strip()
            price_gbp = float(price_text)
            rating = RATING_MAP[book["star_rating"]]
            availability_text = book["availability"].lower()

            if availability_text not in {"in stock", "out of stock"}:
                raise ValueError(f"Unexpected availability: {book['availability']}")

            price_inr = float(
                (Decimal(str(price_gbp)) * Decimal(str(GBP_TO_INR))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP ) )
            cleaned_books.append(
                {
                    "category": book["category"],
                    "title": book["title"],
                    "price_gbp": price_gbp,
                    "price_inr": price_inr,
                    "rating": rating,
                    "in_stock": availability_text == "in stock",
                }
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            dropped_rows.append(
                {"title": book.get("title", "Unknown"), "reason": str(error)}
            )

    clean_df = pd.DataFrame(cleaned_books).astype(
        {
            "category": "string",
            "title": "string",
            "price_gbp": "float64",
            "price_inr": "float64",
            "rating": "int64",
            "in_stock": "bool",
        }
    )
    return clean_df, pd.DataFrame(dropped_rows, columns=["title", "reason"])


def create_and_load_database(clean_df):
    """Create normalized tables and load the freshly scraped catalog snapshot."""
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT UNIQUE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS books (
                book_id INTEGER PRIMARY KEY,
                title TEXT,
                price_gbp REAL,
                price_inr REAL,
                rating INTEGER,
                in_stock INTEGER,
                category_id INTEGER REFERENCES categories(category_id)
            )
        """)

        # This database represents the current scrape snapshot. Delete child rows
        # before parent rows so the foreign-key relationship remains valid.
        cursor.execute("DELETE FROM books")
        cursor.execute("DELETE FROM categories")

        for category_name in clean_df["category"].unique():
            cursor.execute(
                "INSERT INTO categories (category_name) VALUES (?)",
                (category_name,),
            )

        category_id_by_name = dict(
            cursor.execute("SELECT category_name, category_id FROM categories")
        )
        rows = [
            (
                row.title,
                row.price_gbp,
                row.price_inr,
                row.rating,
                int(row.in_stock),
                category_id_by_name[row.category],
            )
            for row in clean_df.itertuples(index=False)
        ]
        cursor.executemany("""
            INSERT INTO books (
                title, price_gbp, price_inr, rating, in_stock, category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)


QUERIES = {
    "01_in_stock_above_20": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1 AND price_gbp > 20
        ORDER BY price_gbp DESC
    """,
    "02_top_10_expensive": """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
    """,
    "03_distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
    """,
    "04_high_rated_books": """
        SELECT title, rating, price_gbp
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, price_gbp DESC
    """,
    "05_top_10_rated_per_category": """
        WITH ranked_books AS (
            SELECT
                b.book_id, b.title, b.price_gbp, b.rating, b.in_stock,
                b.category_id,
                ROW_NUMBER() OVER (
                    PARTITION BY b.category_id
                    ORDER BY b.rating DESC, b.price_gbp DESC, b.title ASC
                ) AS rank_in_category
            FROM books b
        )
        SELECT
            c.category_name, rb.title, rb.price_gbp, rb.rating,
            rb.in_stock, rb.rank_in_category
        FROM ranked_books rb
        JOIN categories c ON rb.category_id = c.category_id
        WHERE rb.rank_in_category <= 10
        ORDER BY c.category_name, rb.rank_in_category
    """,
}


def print_dataframe(dataframe):
    """Print readable output without shortening saved CSV results."""
    display_df = dataframe.copy()
    if "title" in display_df.columns:
        display_df["title"] = display_df["title"].apply(
            lambda title: f"{title[:50]}..." if len(title) > 50 else title
        )
    print(display_df.astype(str).to_string(index=False, justify="left"))


def run_queries_and_validate():
    """Save query text/results and verify SQL JOIN equals pandas merge."""
    RESULTS_DIR.mkdir(exist_ok=True)

    with sqlite3.connect(DATABASE_FILE) as connection:
        for query_name, query in QUERIES.items():
            result_df = pd.read_sql(query, connection)
            (RESULTS_DIR / f"{query_name}.sql").write_text(
                query.strip() + "\n", encoding="utf-8"
            )
            result_df.to_csv(RESULTS_DIR / f"{query_name}.csv", index=False)

            print(f"\n{'=' * 60}\n{query_name}")
            print_dataframe(result_df)

        # Required: read at least two query results through pd.read_sql.
        in_stock_df = pd.read_sql(QUERIES["01_in_stock_above_20"], connection)
        top_expensive_df = pd.read_sql(QUERIES["02_top_10_expensive"], connection)
        sql_join_df = pd.read_sql(
            QUERIES["05_top_10_rated_per_category"], connection
        ).reset_index(drop=True)

        # Required: reproduce the SQL JOIN from in-memory DataFrames.
        books_df = pd.read_sql("SELECT * FROM books", connection)
        categories_df = pd.read_sql("SELECT * FROM categories", connection)
        pandas_join_df = books_df.merge(categories_df, on="category_id", how="inner")
        pandas_join_df = pandas_join_df.sort_values(
            by=["category_id", "rating", "price_gbp", "title"],
            ascending=[True, False, False, True],
        )
        pandas_join_df["rank_in_category"] = (
            pandas_join_df.groupby("category_id").cumcount() + 1
        )
        pandas_join_df = pandas_join_df[
            pandas_join_df["rank_in_category"] <= 10
        ][
            [
                "category_name", "title", "price_gbp", "rating",
                "in_stock", "rank_in_category",
            ]
        ].sort_values(by=["category_name", "rank_in_category"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(sql_join_df, pandas_join_df, check_dtype=False)

        comparison_df = pd.concat(
            {"SQL JOIN": sql_join_df, "pandas merge": pandas_join_df}, axis=1
        )
        comparison_df.to_csv(RESULTS_DIR / "sql_join_vs_pandas_merge.csv", index=False)

        print("\nSQL JOIN result and pandas merge result are equivalent.")
        print("\nFirst pd.read_sql result:")
        print_dataframe(in_stock_df.head())
        print("\nSecond pd.read_sql result:")
        print_dataframe(top_expensive_df.head())


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (MasaiCapstone data pipeline)"})

    scraped_books = scrape_books(session)
    clean_df, dropped_df = clean_books(scraped_books)
    if len(clean_df) < 60:
        raise RuntimeError(f"Expected at least 60 clean books, found {len(clean_df)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(scraped_books).to_csv(RESULTS_DIR / "scraped_books.csv", index=False)
    clean_df.to_csv(RESULTS_DIR / "cleaned_books.csv", index=False)
    dropped_df.to_csv(RESULTS_DIR / "dropped_rows.csv", index=False)

    create_and_load_database(clean_df)
    run_queries_and_validate()

    print(
        f"\nCompleted: {len(clean_df)} clean books loaded from "
        f"{clean_df['category'].nunique()} categories."
    )
    print(f"Database: {DATABASE_FILE}")
    print(f"Artifacts: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
