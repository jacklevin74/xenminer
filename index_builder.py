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

# Fetch all records from the 'blockchain' table
cursor.execute("SELECT records_json FROM blockchain")
records = cursor.fetchall()

# Placeholder for batch inserts
all_values = []

# Loop through each record and prepare data for batch insertion into the 'blocks' table
for record in records:
    records_json = record[0]
    records_list = json.loads(records_json)  # Assuming records_json is in JSON format

    for item in records_list:
        hash_to_verify = item.get("hash_to_verify")
        key = item.get("key")
        account = item.get("account")
        created_at = item.get("date")
        all_values.append((hash_to_verify, key, account, created_at))

# Batch insert into 'blocks' table
try:
    cursor.executemany("""
        REPLACE INTO blocks (hash_to_verify, key, account, created_at)
        VALUES (?, ?, ?, ?)
    """, all_values)
    conn.commit()
except sqlite3.IntegrityError as e:
    print(f"Integrity Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

print(f"Processed {len(all_values)} records.")

# Close the database connection
conn.close()
