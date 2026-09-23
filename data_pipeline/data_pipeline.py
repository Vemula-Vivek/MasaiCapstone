import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sqlite3

# Fetching the 60 books.
books_url = "https://books.toscrape.com/"
books_limit = 60
books=[]
response = requests.get(books_url)
soup = BeautifulSoup(response.text, 'html.parser')

categories = ["Classics","Philosophy","Fiction"]
for category in categories:
    pass
category_links={}

for link in soup.select("div.side_categories ul.nav-list ul a"):
    site_category_name = link.get_text(strip=True)
    if site_category_name in categories:
        category_links[site_category_name] = urljoin(books_url, link["href"])

print(category_links)
for category_name, page_url in category_links.items():
    while page_url and len(books) < books_limit:
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        for book in soup.select("article.product_pod"):
            if len(books) >= books_limit:
                break
            title_tag = book.select_one("h3 a")
            price_tag = book.select_one("p.price_color")
            rating_tag = book.select_one("p.star-rating")
            availability_tag = book.select_one("p.availability")

            books.append(
                {
                    "category": category_name,
                    "title": title_tag["title"],
                    "price_gbp": price_tag.get_text(strip=True),
                    "star_rating": " ".join(rating_tag.get("class", [])),
                    "availability": availability_tag.get_text(" ", strip=True)
                }
            )
        next_link = soup.select_one("li.next a")

        if next_link:
            page_url = urljoin(page_url, next_link["href"])
        else:
            page_url = None

# scraped files cleaning, updating the price, rating
def get_price_gbp(price):
    return price.replace("Â£","")

def get_price_inr(price):
    price_inr = float(price)*105.5
    return price_inr

def get_rating(star_rating):
    rating = star_rating.replace("star-rating ","").lower()
    rating_map = { "one": 1, "two": 2, "three": 3, "four": 4, "five": 5 }
    return rating_map.get(rating)

def get_availability(availability):
    if availability.lower() == "in stock":
        return True
    else:
        return False

for book in books:
    book["price_gbp"] = get_price_gbp(book["price_gbp"])
    book["price_inr"] = get_price_inr(book["price_gbp"])
    book["rating"] = get_rating(book["star_rating"])
    book["in_stock"] = get_availability(book["availability"])
    # del book["star_rating"]


for book in books:
    print(book)

# Designing SQL query for two tables
# Design a normalized SQLite schema with at least two tables sharing a primary/foreign key relationship, for example:
# categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
# books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))

with sqlite3.connect("books.db") as conn:
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    #Table 1
    cursor.execute("""CREATE TABLE IF NOT EXISTS categories (category_id INTEGER PRIMARY KEY, 
    category_name TEXT UNIQUE) """)

    #Table 2
    cursor.execute("""CREATE TABLE IF NOT EXISTS books (book_id INTEGER PRIMARY KEY, title TEXT, 
    price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER, 
    FOREIGN KEY (category_id) REFERENCES categories(category_id))""")

    for book in books:
        category_name = book["category"]

        # Add the category only if it does not already exist.
        cursor.execute("""INSERT OR IGNORE INTO categories (category_name) VALUES (?)""",
                       (category_name,))

        # Obtain its category_id for the books foreign key.
        cursor.execute("""SELECT category_id FROM categories WHERE category_name = ?
        """, (category_name,))

        category_id = cursor.fetchone()[0]

        price_gbp = book["price_gbp"]
        price_inr = book["price_inr"]
        rating = book["rating"]
        in_stock = book["in_stock"]

        cursor.execute("""
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id ) 
        VALUES (?, ?, ?, ?, ?, ?) """, ( book["title"], price_gbp, price_inr, rating,
                                         in_stock, category_id, ))

    conn.commit()

print("Data inserted into books.db successfully.")

with sqlite3.connect("books.db") as connection:
    cursor = connection.cursor()

    cursor.execute("""
        SELECT b.book_id, c.category_name, b.title, b.price_gbp, b.price_inr, b.rating, b.in_stock
        FROM books b JOIN categories c ON b.category_id = c.category_id """)

    for row in cursor.fetchall():
        print(row)

# Deleting the tables in the database