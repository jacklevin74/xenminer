import re
import sqlite3
import threading
import time

def count_uppercase_letters(hash_to_verify):
    capital_count = 0
    for char in hash_to_verify:
        if char.isalpha() and char.isupper():
            capital_count += 1
    return capital_count

def run_db_operations():
    max_retries = 3
    for i in range(max_retries):
        try:
            # Connect to SQLite database
            conn = sqlite3.connect('blockchain.db', timeout=10)  # 10 seconds timeout
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
            cursor.execute("""
            SELECT hash_to_verify, account FROM blocks;
            """)

            rows = cursor.fetchall()

            for row in rows:
                hash_to_verify, account_to_update = row
                capital_count = count_uppercase_letters(hash_to_verify)
                print ("Scanning account: ", account_to_update, capital_count)

                if capital_count >= 65:
                    print("Found superblock for: ", account_to_update, capital_count)
                    # Update the super_block_count of the account in the super_blocks table
                    cursor.execute("""
                    UPDATE super_blocks
                    SET super_block_count = super_block_count + 1
                    WHERE account = ?;
                    """, (account_to_update,))

                    if cursor.rowcount == 0:
                        # The account doesn't exist, insert a new row
                        cursor.execute("""
                        INSERT INTO super_blocks (account, super_block_count)
                        VALUES (?, ?);
                        """, (account_to_update, 1))

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
    #threading.Timer(300, run_db_operations).start()

# Kick off the first run
run_db_operations()
