import requests
import json
import hashlib
import argparse
from passlib.hash import argon2

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

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='XenMiner PoW submission')
parser.add_argument('account_address', type=str, help='The account address (like 0x...) for submitting the PoW')
args = parser.parse_args()
account_address = args.account_address

# Download last block record
url = 'http://xenminer.mooo.com:4445/getblocks/lastblock'
response = requests.get(url)

if response.status_code == 200:
    records = json.loads(response.text)

    verified_hashes = []
    for record in records:
        block_id = record.get('block_id')
        hash_to_verify = record.get('hash_to_verify')
        key = record.get('key')
        account = record.get('account')
        
        # Verify each record using Argon2
        if argon2.verify(key, hash_to_verify):
            verified_hashes.append(hash_value(str(block_id) + hash_to_verify + key + account))

    # If we have any verified hashes, build the Merkle root
    if verified_hashes:
        merkle_root, _ = build_merkle_tree(verified_hashes)

        # Calculate block ID for output (using the last record for reference)
        output_block_id = int(block_id / 100)
        
        # Prepare payload for PoW
        payload = {
            'account_address': account_address,
            'block_id': output_block_id,
            'merkle_root': merkle_root
        }
        
        # Send POST request
        pow_response = requests.post('http://xenminer.mooo.com:4446/send_pow', json=payload)
        
        if pow_response.status_code == 200:
            print(f"Proof of Work successful: {pow_response.json()}")
        else:
            print(f"Proof of Work failed: {pow_response.json()}")

        print(f"Block ID: {output_block_id}, Merkle Root: {merkle_root}")

else:
    print("Failed to fetch the last block.")

