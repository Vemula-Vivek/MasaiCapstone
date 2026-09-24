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
