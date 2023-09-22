import re
import sqlite3
import threading
import time
from collections import defaultdict

def run_db_operations():
    max_retries = 3
    for i in range(max_retries):
        try:
            # Connect to SQLite database
            conn = sqlite3.connect('blocks.db', timeout=10)  # 10 seconds timeout
            cursor = conn.cursor()

            # Drop super_blocks table if it exists
            cursor.execute("DROP TABLE IF EXISTS super_blocks")

            # Create the table
            cursor.execute("""
            CREATE TABLE super_blocks (
                account TEXT PRIMARY KEY,
                super_block_count INTEGER
            );
            """)

            # Fetch all hash_to_verify and account records from blocks table
            cursor.execute("SELECT hash_to_verify, account FROM blocks;")
            rows = cursor.fetchall()

            # Prepare a dictionary to keep counts
            super_block_counts = defaultdict(int)

            for row in rows:
                hash_to_verify, account_to_update = row
                capital_count = sum(1 for char in re.sub('[0-9]', '', hash_to_verify) if char.isupper())

                if capital_count >= 65:
                    super_block_counts[account_to_update] += 1

            # Insert all rows in one go
            cursor.executemany("""
            INSERT INTO super_blocks (account, super_block_count)
            VALUES (?, ?);
            """, super_block_counts.items())

            # Commit the changes to the database
            conn.commit()

            # Close the connection
            conn.close()

            # If successful, break the retry loop
            break
        except sqlite3.OperationalError:
            print(f"Database is locked, retrying {i+1}/{max_retries}")
            time.sleep(1)  # wait for 1 second before retrying

    # Schedule the next run
    threading.Timer(300, run_db_operations).start()

# Kick off the first run
run_db_operations()

