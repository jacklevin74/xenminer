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
            super_blocks INTEGER
        )""")
        cache_conn.commit()

        try:
            cache_cursor.execute("""
            ALTER TABLE cache_table ADD COLUMN rank INTEGER DEFAULT 0
            """)
            cache_conn.commit()
        except sqlite3.OperationalError:
            # Ignore error, if the column already exists.
            pass

        # Fetch data from the original database and populate the cache table
        original_cursor.execute("""
        WITH RankedBlocks AS (
            SELECT 
                b.account,
                COUNT(b.block_id) AS total_blocks,
                100000 AS hashes_per_second,  -- static value of 100000 for hashes_per_second
                COALESCE(sb.super_block_count, 0) AS super_blocks
            FROM blocks b
            LEFT JOIN super_blocks sb USING (account)
            GROUP BY b.account
        )
        SELECT *,
               ROW_NUMBER() OVER (ORDER BY total_blocks DESC) AS rank
        FROM RankedBlocks
        ORDER BY rank;
        """)

        rows = original_cursor.fetchall()
        num_rows_fetched = len(rows)  # Get the number of rows fetched

        # Insert fetched rows into the cache table
        cache_cursor.executemany("""
        INSERT OR REPLACE INTO cache_table (account, total_blocks, hashes_per_second, super_blocks, rank) VALUES (?, ?, ?, ?, ?)
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
