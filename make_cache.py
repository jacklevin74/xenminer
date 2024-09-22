import sqlite3
import time
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

def recreate_cache_table():
    logging.info("Generating cache table")

    try:
        original_conn = sqlite3.connect("blocks.db")
        cache_conn = sqlite3.connect("cache.db")
        original_cursor = original_conn.cursor()
        cache_cursor = cache_conn.cursor()

        # Create the new cache table
        cache_cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_table (
            account TEXT PRIMARY KEY,
            total_blocks INTEGER,
            hashes_per_second REAL,
            super_blocks INTEGER,
            rank INTEGER DEFAULT 0,
            xnm BIGINT DEFAULT 0,
            xblk BIGINT DEFAULT 0,
            xuni BIGINT DEFAULT 0
        )""")
        cache_conn.commit()

        try:
            cache_cursor.execute("""
            ALTER TABLE cache_table ADD COLUMN rank INTEGER DEFAULT 0
            """)
        except sqlite3.OperationalError:
            pass

        try:
            cache_cursor.execute("""
            ALTER TABLE cache_table ADD COLUMN xnm BIGINT DEFAULT 0
            """)
        except sqlite3.OperationalError:
            pass

        try:
            cache_cursor.execute("""
            ALTER TABLE cache_table ADD COLUMN xblk BIGINT DEFAULT 0
            """)
        except sqlite3.OperationalError:
            pass

        try:
            cache_cursor.execute("""
            ALTER TABLE cache_table ADD COLUMN xuni BIGINT DEFAULT 0
            """)
        except sqlite3.OperationalError:
            pass

        # Fetch data from the original database and populate the cache table
        original_cursor.execute("""
        WITH epoch_calculations AS (
            SELECT
                LOWER(b.account) AS account,
                b.block_id,
                strftime('%Y', b.created_at) AS created_year,
                strftime('%m-%d %H:%M:%S', b.created_at) AS created_time
            FROM blocks b
        ),
        
        grouped_blocks_by_epoch AS (
            SELECT
                LOWER(b.account) AS account,
                1 + CAST(
                        (created_year - 2023) +
                        CASE
                            WHEN created_time >= '09-16 21:00:00' THEN 0
                            ELSE -1
                            END AS INTEGER
                    ) AS epoch,
                COUNT(b.block_id) AS blocks_per_epoch
            FROM epoch_calculations b
            GROUP BY 1, 2
        ),
        
        xuni_counts AS (
              SELECT
                  LOWER(account) AS account,
                  COUNT(*) AS total_xuni
              FROM xuni
              GROUP BY 1
        ),
        
        account_performance as (SELECT
            b.account,
            SUM(blocks_per_epoch) AS total_blocks,
            SUM(blocks_per_epoch * POWER(10, 19) / POWER(2, epoch - 1)) AS xnm,
            COALESCE(sb.super_block_count, 0) AS super_blocks,
            COALESCE(x.total_xuni, 0) AS total_xuni
        FROM grouped_blocks_by_epoch b
            LEFT JOIN super_blocks sb ON b.account = sb.account
            LEFT JOIN xuni_counts x ON b.account = x.account
        GROUP BY b.account)
        
        SELECT
            account,
            ROW_NUMBER() OVER (ORDER BY total_blocks DESC, super_blocks DESC, total_xuni DESC, account DESC) AS rank,
            total_blocks,
            super_blocks,
            xnm,
            super_blocks * power(10, 18) AS xblk,
            total_xuni * power(10, 18) AS xuni,
            100000 AS hashes_per_second
        FROM account_performance
        ORDER BY rank
        """)

        rows = original_cursor.fetchall()
        num_rows_fetched = len(rows)  # Get the number of rows fetched
        logging.info(f"Fetched {num_rows_fetched} rows from the original database.")

        # Insert fetched rows into the cache table
        cache_cursor.executemany("""
        INSERT OR REPLACE INTO cache_table 
            (account, rank, total_blocks, super_blocks, xnm, xblk, xuni, hashes_per_second) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        cache_conn.commit()

        # Close the database connections
        original_conn.close()
        cache_conn.close()
        logging.info(f"Cache table updated. {num_rows_fetched} rows were fetched and inserted.")

    except sqlite3.OperationalError as e:
        logging.error(f"SQLite error: {e}. Retrying in 5 seconds.")


def run_db_operations():
    while True:
        recreate_cache_table()
        time.sleep(300)


if __name__ == "__main__":
    logging.info("Starting make_cache")
    t = threading.Thread(target=run_db_operations, daemon=True)
    t.start()
    t.join()
