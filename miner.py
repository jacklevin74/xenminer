import json
import requests
import time
from passlib.hash import argon2
import hashlib
from random import choice, randrange
import string


difficulty = 1
memory_cost = 80
cores = 1
account = "0x0A6969ffF003B760c97005e03ff5a9741126167A"

class Block:
    def __init__(self, index, prev_hash, data, valid_hash, random_data, attempts):
        self.index = index
        self.prev_hash = prev_hash
        self.data = data
        self.valid_hash = valid_hash
        self.random_data = random_data
        self.attempts = attempts
        self.timestamp = time.time()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        sha256 = hashlib.sha256()
        sha256.update(f"{self.index}{self.prev_hash}{self.data}{self.valid_hash}{self.timestamp}".encode("utf-8"))
        return sha256.hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "data": self.data,
            "valid_hash": self.valid_hash,
            "random_data": self.random_data,
            "timestamp": self.timestamp,
            "hash": self.hash,
            "attempts": self.attempts
        }

def generate_random_sha256(max_length=128):
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(choice(characters) for _ in range(randrange(1, max_length + 1)))

    sha256 = hashlib.sha256()
    sha256.update(random_string.encode('utf-8'))
    return sha256.hexdigest()

def mine_block(target_substr, prev_hash):
    argon2_hasher = argon2.using(time_cost=difficulty, salt=b"XEN10082022XEN", memory_cost=memory_cost, parallelism=cores, hash_len = 64)
    attempts = 0
    random_data = None
    start_time = time.time()

    while True:
        attempts += 1
        random_data = generate_random_sha256()
        hashed_data = argon2_hasher.hash(random_data + prev_hash)

        if target_substr in hashed_data[-87:]:
            print(f"Found valid hash after {attempts} attempts: {hashed_data}")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    hashes_per_second = attempts / elapsed_time

    # Prepare the payload
    payload = {
        "hash_to_verify": hashed_data,
        "key": random_data + prev_hash,
        "account": account,
        "attempts": attempts,
        "hashes_per_second": hashes_per_second
        }

    print (payload)

    # Make the POST request
    response = requests.post('http://proofofwork.mooo.com/verify', json=payload)

    # Print the HTTP status code
    print("HTTP Status Code:", response.status_code)

    # Print the server's response
    try:
        print("Server Response:", response.json())
    except Exception as e:
        print("An error occurred:", e)


    return random_data, hashed_data, attempts, hashes_per_second

def verify_block(block):
    argon2_hasher = argon2.using(time_cost=difficulty, memory_cost=memory_cost, parallelism=cores)
    #debug
    print ("Key: ");
    print (block['random_data'] + block['prev_hash'])
    print ("Hash: ");
    print (block['valid_hash'])
    return argon2_hasher.verify(block['random_data'] + block['prev_hash'], block['valid_hash'])

if __name__ == "__main__":
    blockchain = []
    target_substr = "XEN11"
    num_blocks_to_mine = 20000000

    genesis_block = Block(0, "0", "Genesis Block", "0", "0", "0")
    blockchain.append(genesis_block.to_dict())
    print(f"Genesis Block: {genesis_block.hash}")

    for i in range(1, num_blocks_to_mine + 1):
        print(f"Mining block {i}...")
        random_data, new_valid_hash, attempts, hashes_per_second = mine_block(target_substr, blockchain[-1]['hash'])
        new_block = Block(i, blockchain[-1]['hash'], f"Block {i} Data", new_valid_hash, random_data, attempts)
        new_block.to_dict()['hashes_per_second'] = hashes_per_second
        blockchain.append(new_block.to_dict())
        print(f"New Block Added: {new_block.hash}")

    # Verification
    for i, block in enumerate(blockchain[1:], 1):
        is_valid = verify_block(block)
        print(f"Verification for Block {i}: {is_valid}")

    # Write blockchain to JSON file
    blockchain_json = json.dumps(blockchain, indent=4)
    with open("blockchain.json", "w") as f:
        f.write(blockchain_json)

