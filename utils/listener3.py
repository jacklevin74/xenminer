import asyncio
import websockets
import zlib
from datetime import datetime
import time
import hashlib
import sqlite3
import json
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from collections import deque
import hashlib
import concurrent.futures

def compute_sha256(data):
    return hashlib.sha256(data.encode()).hexdigest()

# Initialize a deque with a fixed size of 10
recent_messages = deque(maxlen=10)
recent_blocks = deque(maxlen=30)
response_queue = asyncio.Queue()

DATABASE_NAME = 'blocks.db'
ready_flag = False

import threading
import time
from passlib.hash import argon2
from hashlib import sha256


# Function to verify the hash with Argon2id
def verify_argon2id_hash(hashed_data, key):
    ph = PasswordHasher()
    try:
        ph.verify(hashed_data, key)
        return True
    except exceptions.VerifyMismatchError:
        return False

# Function to process a set of blocks
def process_block_set(blocks):
    start_time = time.time()
    current_hash = ""
    verify_results = []

    # Function to handle verification in a separate thread
    def verify_and_append(block):
        _, hash_to_verify, key, *rest = block
        verification_result = verify_argon2id_hash(hash_to_verify, key)
        verify_results.append((block[0], verification_result))

    threads = []
    for block in blocks:
        block_str = '|'.join(block)
        thread = threading.Thread(target=verify_and_append, args=(block,))
        threads.append(thread)
        thread.start()

        current_hash = sha256((current_hash + block_str).encode()).hexdigest()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000

    return current_hash, elapsed_time_ms, verify_results

# Async function to watch recent blocks
async def watch_recent_blocks():
    processed_ids = set()  # Keeps track of processed block_ids
    blocks_to_process = deque(maxlen=10)  # Holds up to 10 most recent blocks

    while True:
        for block_data in recent_blocks:
            block_id_str = block_data.split('|')[0]
            if block_id_str not in processed_ids:
                processed_ids.add(block_id_str)
                blocks_to_process.append(block_data.split('|'))  # Add the block to the processing queue

                if block_id_str.endswith('0') and len(blocks_to_process) >= 10:
                    block_list = list(blocks_to_process)[-10:]
                    final_hash, elapsed_time_ms, verify_results = process_block_set(block_list)
                    print(f"Final hash for block set starting with {block_id_str}: {final_hash}, processing time: {elapsed_time_ms} ms")
                    for block_id, result in verify_results:
                        print(f"Block {block_id} verification: {'Success' if result else 'Failure'}")
                    blocks_to_process.clear()  # Clear the deque for the next set of blocks

        await asyncio.sleep(1)  # Check interval (e.g., every 1 second


# Function to store the message in the deque
def store_message(message):
    recent_messages.append(message)

# Initialize the database
def init_db():
    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS "blocks" (
                block_id INTEGER PRIMARY KEY,
                hash_to_verify TEXT,
                key TEXT UNIQUE,
                account TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

# Compute truncated SHA256 hash
def compute_truncated_sha256(s):
    sha256_hash = hashlib.sha256(s.encode()).hexdigest()
    truncated_hash = sha256_hash[:5]
    return truncated_hash

# Function to verify the hash with Argon2
def verify_argon2id_hash(hashed_data, key):
    ph = PasswordHasher()
    try:
        ph.verify(hashed_data, key)
        return True
    except VerifyMismatchError:
        return False

async def send_responses(websocket):
    while True:
        response_data = await response_queue.get()
        await websocket.send(response_data)
        response_queue.task_done()


# Process the received data
async def process_data(message):
    decompressed_data = zlib.decompress(message).decode('utf-8')

    parts = decompressed_data.split('|')

    block_id, hash_to_verify, key, account, created_at, timestamp_str = parts
    # Verify the hash with Argon2
    #is_valid_hash = verify_argon2id_hash(hash_to_verify, key)
    #if not is_valid_hash:
    #    print(f"Invalid hash for block ID: {block_id}")
    #    return  # Or handle invalid hash case as needed

    timestamp_int = int(timestamp_str)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    local_timestamp = int(time.time() * 1000)
    timestamp_diff = local_timestamp - timestamp_int
    hash = compute_truncated_sha256(decompressed_data)

    store_message(decompressed_data)
    recent_blocks.append(decompressed_data)

    # Send response back through the same WebSocket
    response_data = {"block_id": block_id, "hash": hash, "time_diff": timestamp_diff}
    await response_queue.put(json.dumps(response_data))

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + " diff: " + str(timestamp_diff) + f" ms block_id: {block_id} " + hash)

    with open("websocket_data.txt", "a") as file:
        file.write(f"{timestamp}: {decompressed_data}\n")

    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO blocks (block_id, hash_to_verify, key, account, created_at) VALUES (?, ?, ?, ?, ?)",
            (block_id, hash_to_verify, key, account, created_at)
        )
        conn.commit()


# WebSocket reader coroutine
async def websocket_reader(websocket):
    global ready_flag
    pong_count = 0

    while True:
        message = await websocket.recv()
        await process_data(message)


async def echo(websocket, path):
    print ("Starting echo server")
    async for message in websocket:
        if message == "request":
            # Convert deque to a list and then to a JSON string
            json_data = json.dumps(list(recent_messages))
            await websocket.send(json_data)


# Start the server
async def start_server():
    async with websockets.serve(echo, "0.0.0.0", 8765):  # Use your desired port here
        await asyncio.Future()  # This will run forever

# Main coroutine
async def main():
    global ready_flag
    server_task = asyncio.create_task(start_server())
    watch_task = asyncio.create_task(watch_recent_blocks())
    while True:
        pong_count = 0      # Reset the pong count as well
        try:
            async with websockets.connect('ws://xenblocks.io:6668') as websocket:
                print("Connected to the server!")
                reader_task = asyncio.create_task(websocket_reader(websocket))
                sender_task = asyncio.create_task(send_responses(websocket))

                await asyncio.gather(reader_task, sender_task, server_task, watch_task)
        except Exception as e:
            ready_flag = False  # Reset the ready flag each time before connecting
            print(f"Connection error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


# Initialize the database
init_db()

# Run the main coroutine
asyncio.get_event_loop().run_until_complete(main())

