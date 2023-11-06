import sqlite3
import os
import shutil
import time


def recreate_cache_table():
    while True:  # Run indefinitely
        try:
            # Copy the original database to /tmp directory
            original_db_path = "blocks.db"
            tmp_db_path = "/tmp/blocks_copy.db"
            #shutil.copy(original_db_path, tmp_db_path)

            # Connect to the copied and cache databases
            original_conn = sqlite3.connect(original_db_path)
            cache_conn = sqlite3.connect("cache.db")
            # Connect to the original and cache databases
            #original_conn = sqlite3.connect("blocks.db")
            #cache_conn = sqlite3.connect("cache.db")

            # Create cursors for each database
            original_cursor = original_conn.cursor()
            cache_cursor = cache_conn.cursor()

            # Drop the existing cache table if it exists
            # cache_cursor.execute("DROP TABLE IF EXISTS cache_table")

            # Create the new cache table
            cache_cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_table (
                account TEXT PRIMARY KEY,
                total_blocks INTEGER,
                hashes_per_second REAL,
                super_blocks INTEGER
            )""")
            cache_conn.commit()

            # Fetch data from the original database and populate the cache table
            original_cursor.execute("""
            SELECT b.account,
                COUNT(b.block_id) AS total_blocks,
                100000 AS hashes_per_second,  -- static value of 100000 for hashes_per_second
                COALESCE(sb.super_block_count, 0) AS super_blocks
            FROM blocks b
            LEFT JOIN (SELECT account, super_block_count FROM super_blocks) sb ON b.account = sb.account
            GROUP BY b.account
            ORDER BY total_blocks DESC
            """)

            rows = original_cursor.fetchall()
            num_rows_fetched = len(rows)  # Get the number of rows fetched

            # Insert fetched rows into the cache table
            cache_cursor.executemany("REPLACE INTO cache_table (account, total_blocks, hashes_per_second, super_blocks) VALUES (?, ?, ?, ?)", rows)
            cache_conn.commit()


            # Close the database connections
            original_conn.close()
            cache_conn.close()
            print(f"Cache table updated. {num_rows_fetched} rows were fetched and inserted.")

            # Wait for 5 minutes before the next iteration
            time.sleep(300)

        except sqlite3.OperationalError as e:
            print(f"SQLite error: {e}. Retrying in 5 seconds.")
            time.sleep(5)  # Wait for 5 seconds before retrying

if __name__ == "__main__":
    recreate_cache_table()

