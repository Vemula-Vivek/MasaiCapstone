SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1 AND price_gbp > 20
        ORDER BY price_gbp DESC
