import re
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('blocks.db')
cursor = conn.cursor()

# Fetch block_id, hash_to_verify, and account records from blocks table
cursor.execute("""
SELECT block_id, hash_to_verify, account, created_at FROM blocks;
""")

rows = cursor.fetchall()

for row in rows:
    block_id, hash_to_verify, account, created_at = row
    #capital_count = sum(1 for char in re.sub('[0-9]', '', hash_to_verify) if char.isupper())
    last_element = hash_to_verify.split("$")[-1]
    hash_uppercase_only = ''.join(filter(str.isupper, last_element))
    capital_count = len(hash_uppercase_only)


    if capital_count >= 50:
        print(f"Block ID: {hash_to_verify}, {block_id} {created_at}, Capital Count: {capital_count}, Account: {account}")

# Close the connection
conn.close()
