SELECT title, rating, price_gbp
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, price_gbp DESC
