import sqlite3
import json
from datetime import datetime

# Connect to the SQLite database
conn = sqlite3.connect("blockchain.db")
cursor = conn.cursor()

# Create the 'blocks' table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS blocks (
        block_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hash_to_verify TEXT,
        key TEXT UNIQUE,
        account TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

# Fetch all records from the 'blockchain' table
cursor.execute("SELECT records_json FROM blockchain")
records = cursor.fetchall()

# Initialize counter for periodic progress check
counter = 0

# Loop through each record and insert it into the 'blocks' table
for record in records:
    records_json = record[0]
    records_list = json.loads(records_json)  # Assuming records_json is in JSON format

    for item in records_list:
        block_id = item.get("block_id")
        hash_to_verify = item.get("hash_to_verify")
        key = item.get("key")
        account = item.get("account")
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            cursor.execute("""
                REPLACE INTO blocks (hash_to_verify, key, account, created_at)
                VALUES (?, ?, ?, ?)
            """, (hash_to_verify, key, account, created_at))
            conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Integrity Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Increment the counter and check for progress
        counter += 1
        if counter % 10000 == 0:
            print(f"Processed {counter} records so far.")

# Close the database connection
conn.close()
