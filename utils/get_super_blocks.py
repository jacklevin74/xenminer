import sqlite3
import json
import re
from collections import defaultdict

def generate_superblock_report(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    cursor = conn.cursor()

    # Prepare a dictionary to keep counts
    super_blocks_per_block = defaultdict(lambda: defaultdict(int))

    # Fetch the first row from the blockchain table
    cursor.execute("SELECT id, records_json FROM blockchain LIMIT 1;")
    block = cursor.fetchone()

    while block is not None:
        block_id = block['id']
        records_json = block['records_json']
        try:
            # Parse the JSON data
            records = json.loads(records_json)

            # Analyze each record for superblocks
            for record in records:
                hash_to_verify = record.get('hash_to_verify')
                account_to_update = record.get('account')
                if hash_to_verify and 'WEVOMTAwODIwMjJYRU4' in hash_to_verify and ('XEN' in hash_to_verify or 'XEN1' in hash_to_verify or 'XEN11' in hash_to_verify):
                    capital_count = sum(1 for char in re.sub('[0-9]', '', hash_to_verify) if char.isupper())
                    if capital_count >= 65:
                        #print(f"Block ID: {block_id}, Account: {account_to_update}, Hash: {hash_to_verify}")
                        super_blocks_per_block[block_id][account_to_update] += 1
        except json.JSONDecodeError:
            print(f"Error decoding JSON for block ID: {block_id}")

        # Fetch the next row
        cursor.execute("SELECT id, records_json FROM blockchain WHERE id > ? ORDER BY id ASC LIMIT 1;", (block_id,))
        block = cursor.fetchone()

    # Close the database connection
    conn.close()

    # Print the report
    for block_id, accounts in super_blocks_per_block.items():
        for account, superblock_count in accounts.items():
            print(f"Block ID: {block_id}, Account: {account}, Superblock Balance: {superblock_count}")

# Replace 'your_database_path.db' with the path to your actual database
generate_superblock_report('blockchain.db')

