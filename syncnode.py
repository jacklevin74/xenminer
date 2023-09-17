import sqlite3
import argparse
import requests
from passlib.hash import argon2
import hashlib
import json


# Create the parser
parser = argparse.ArgumentParser(description="Your script description")

# Add Ethereum address argument
parser.add_argument("ethereum_address", type=str, help="Your Ethereum address")

# Parse the arguments
args = parser.parse_args()

# Set the Ethereum address as a global variable
global my_ethereum_address
my_ethereum_address = args.ethereum_address



def hash_value(value):
    return hashlib.sha256(value.encode()).hexdigest()

def build_merkle_tree(elements, merkle_tree={}):
    if len(elements) == 1:
        return elements[0], merkle_tree

    new_elements = []
    for i in range(0, len(elements), 2):
        left = elements[i]
        right = elements[i + 1] if i + 1 < len(elements) else left
        combined = left + right
        new_hash = hash_value(combined)
        merkle_tree[new_hash] = {'left': left, 'right': right}
        new_elements.append(new_hash)
    return build_merkle_tree(new_elements, merkle_tree)


def get_total_blocks():
    # Send a GET request to retrieve the JSON response
    url = "http://xenminer.mooo.com/total_blocks"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        try:
            # Parse the JSON response
            data = json.loads(response.text)
            total_blocks_top100 = data.get("total_blocks_top100", 0)

            # Subtract 100 and divide by 100 without remainder
            adjusted_value = (total_blocks_top100 - 100) // 100
            return adjusted_value
        except Exception as e:
            print(f"Error parsing JSON: {e}")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

    return None

conn = sqlite3.connect('blockchain.db')
c = conn.cursor()

# Create blockchain table
c.execute('''CREATE TABLE IF NOT EXISTS blockchain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    prev_hash TEXT,
                    merkle_root TEXT,
                    records_json TEXT,
                    block_hash TEXT)''')
# Fetch the latest block ID from the blockchain
c.execute('SELECT MAX(id) FROM blockchain')
row = c.fetchone()
last_block_id = row[0] if row and row[0] is not None else 0
print ("Last fetched block ID from blockchain: ", last_block_id)

# Get the total blocks from the API
total_blocks = get_total_blocks()
if total_blocks is None:
    print("Failed to retrieve total_blocks.")
else:
    # Calculate the difference between total_blocks and last_block_id
    difference = total_blocks - last_block_id

    # Define how many records to fetch based on the difference
    num_to_fetch = difference if difference > 0 else 1  # Ensure at least 1 record is fetched

    # Calculate the end_block_id
    end_block_id = last_block_id + num_to_fetch
    print("Number of records to fetch:", num_to_fetch)
    print("End block ID:", end_block_id)

# Fetch the latest block_hash from the blockchain
c.execute('SELECT block_hash FROM blockchain ORDER BY id DESC LIMIT 1')
row = c.fetchone()
prev_hash = row[0] if row else 'genesis'
print ("Found previous record in blockchain, continuing with hash: ", prev_hash)

# Loop through block IDs starting from the last fetched ID
for block_id in range(last_block_id + 1, end_block_id + 1):
    url = f"http://xenminer.mooo.com:4445/getblocks/{block_id}"
    #print ("Fetching URL: ", url)
    response = requests.get(url)
    if response.status_code == 200:
        records = json.loads(response.text)
        verified_hashes = []
        #print ("Fetching block_id ", block_id);
        for record in records:
            hash_to_verify = record.get("hash_to_verify")
            key = record.get("key")
            account = record.get("account")

            if argon2.verify(key, hash_to_verify):
                #print ("Argon2 verified for block_id", block_id);
                verified_hashes.append(hash_value(str(block_id) + hash_to_verify + key + account))

        if verified_hashes:  # Only insert if there are verified hashes
            merkle_root, _ = build_merkle_tree(verified_hashes)
            records_json_blob = json.dumps(records)

            # Generate block hash using timestamp, prev_hash, and merkle_root
            block_contents = str(prev_hash) + str(merkle_root)
            block_hash = hash_value(block_contents)

            # Insert new block into the blockchain table
            c.execute('INSERT INTO blockchain (prev_hash, merkle_root, records_json, block_hash) VALUES (?, ?, ?, ?)',
                      (prev_hash, merkle_root, records_json_blob, block_hash))
            print ("Fetched block with merkleroot ", block_id, merkle_root)
            conn.commit()

            # Set prev_hash for the next iteration
            prev_hash = block_hash

c.execute('SELECT id, id, block_hash FROM blockchain order by id desc limit 1')
row = c.fetchone()
if row:
    total_count, last_block_id, last_block_hash = row
    validation_data = {
            "total_count": total_count,
            "my_ethereum_address": my_ethereum_address,
            "last_block_id": last_block_id,
            "last_block_hash": last_block_hash
            }
    print (validation_data)
    requests.post("http://xenminer.mooo.com/validate", json=validation_data)

conn.close()

def verify_block_hashes():
    conn = sqlite3.connect('blockchain.db')
    c = conn.cursor()
    c.execute('SELECT id, timestamp, prev_hash, merkle_root, block_hash, records_json FROM blockchain ORDER BY id')

    prev_hash = 'genesis'  # Initialize with genesis hash
    for row in c.fetchall():
        id, timestamp, prev_hash_db, merkle_root, block_hash, records_json = row

        # Verify block hash
        block_contents = str(prev_hash) + str(merkle_root)
        computed_block_hash = hash_value(block_contents)
        if computed_block_hash != block_hash:
            print(f"Block {id} is invalid. Computed hash doesn't match the stored hash.")
            return False

        # Verify Merkle root and Argon2 hashes
        records = json.loads(records_json)
        verified_hashes = []
        for record in records:
            hash_to_verify = record.get("hash_to_verify")
            key = record.get("key")
            account = record.get("account")

            if argon2.verify(key, hash_to_verify):
                verified_hashes.append(hash_value(str(id) + hash_to_verify + key + account))
                #print ("Key and hash_to_verify pass argon2 verification ", key, hash_to_verify)
            else:
                print ("Key and hash_to_verify fail argon2 verification ", key, hash_to_verify)
                return False;

        if verified_hashes:
            computed_merkle_root, _ = build_merkle_tree(verified_hashes)
            if computed_merkle_root != merkle_root:
                print(f"Block {id} is invalid. Computed Merkle root doesn't match the stored Merkle root.")
                return False
            else:
                print (f"Block {id} is valid. Computed Merkle root match the stored Merkle root.")

        # Set prev_hash for the next iteration
        prev_hash = block_hash

    print("All blocks are valid.")


    return True

# Call verify_block_hashes after your existing code
#verify_block_hashes()
