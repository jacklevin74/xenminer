import sqlite3
import threading
import time
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s"
)


def update_super_blocks():
    logging.info("Updating super_blocks table")

    max_retries = 3
    for i in range(max_retries):
        try:
            # Connect to SQLite database
            conn = sqlite3.connect("blocks.db", timeout=10)  # 10 seconds timeout
            cursor = conn.cursor()

            # Create the table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS super_blocks (
                account TEXT PRIMARY KEY,
                super_block_count INTEGER
            );
            """
            )

            # Fetch all hash_to_verify and account records from blocks table
            cursor.execute("SELECT hash_to_verify, account FROM blocks")
            rows = cursor.fetchall()

            # Prepare a dictionary to keep counts
            super_block_counts = defaultdict(int)

            for row in rows:
                hash_to_verify, account_to_update = row
                last_element = hash_to_verify.split("$")[-1]
                hash_uppercase_only = "".join(filter(str.isupper, last_element))
                capital_count = len(hash_uppercase_only)

                if capital_count >= 50:
                    super_block_counts[account_to_update] += 1

            # Insert all rows in one go
            cursor.executemany(
                """
            INSERT OR REPLACE INTO super_blocks (account, super_block_count)
            VALUES (?, ?)
            """,
                super_block_counts.items(),
            )

            # Commit the changes to the database
            conn.commit()

            # Close the connection
            conn.close()

            # If successful, break the retry loop
            break
        except sqlite3.OperationalError as e:
            logging.error(f"Database is locked, retrying {i+1}/{max_retries} - {e}")
            time.sleep(1)  # wait for 1 second before retrying

    logging.info("Super_blocks table updated successfully")


def run_db_operations():
    while True:
        update_super_blocks()
        time.sleep(300)


if __name__ == "__main__":
    logging.info("Starting super_blocks")
    t = threading.Thread(target=run_db_operations, daemon=True)
    t.start()
    t.join()
