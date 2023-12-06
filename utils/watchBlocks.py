from web3 import Web3
from passlib.hash import argon2
import struct
import time
import json

# Ethereum connection setup
w3 = Web3(Web3.HTTPProvider("https://x1-testnet.infrafc.org"))
if not w3.is_connected():
    print("Failed to connect to the Ethereum network")
    exit()

# Load contract ABI and address
with open('build/contracts/BlockStorage.json', 'r') as file:
    contract_json = json.load(file)
    contract_abi = contract_json['abi']

contract_address = "0xa21DF1C412ceAE97bf1af79d4eDeCA8c9686EC30"  # Replace with your contract address

contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Event filter for the 'NewHash' event
new_hash_filter = contract.events.NewHash.create_filter(fromBlock='latest')

def generate_argon2_hash(key, m_value, t_value, p_value, salt_bytes):
    # Ensure key and salt_bytes are in bytes format
    if isinstance(key, str):
        key = key.encode()
    if isinstance(salt_bytes, str):
        salt_bytes = salt_bytes.encode()

    # Configure the Argon2 hasher
    argon2_hasher = argon2.using(time_cost=t_value, memory_cost=m_value, parallelism=p_value, salt=salt_bytes, hash_len=64)

    # Generate the hash using the configured hasher
    hashed_data = argon2_hasher.hash(key)

    return hashed_data

def handle_event(event):
    hash_id = event['args']['hashId']  # Adjust according to your event parameters
    decoded_data = contract.functions.decodeRecordBytes(hash_id).call()
    
    c, m, t, v, k, s = decoded_data

    # Use the generate_argon2_hash function
    argon2_hash = generate_argon2_hash(k.hex(), m, t, c, s)

    print(f"Argon2 Hash: {argon2_hash}")



# Poll for new events
def log_loop(event_filter, poll_interval):
    while True:
        for event in event_filter.get_new_entries():
            handle_event(event)
        time.sleep(poll_interval)

# Run the loop
log_loop(new_hash_filter, 2)  # Poll every 2 seconds
