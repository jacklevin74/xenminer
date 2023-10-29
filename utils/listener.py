import asyncio
import websockets
import zlib
import time
import hashlib
import sqlite3
from datetime import datetime

DATABASE_NAME = 'blocks.db'

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

# Global variable for cumulative hash
cumulative_hash = ""
hash_counter = 0

# Process the received data
async def process_data(websocket, message, next_starting_id):
    global cumulative_hash, hash_counter
    decompressed_data = zlib.decompress(message).decode('utf-8')
    parts = decompressed_data.split('|')

    block_id, hash_to_verify, key, account, created_at, timestamp_str = parts
    block_id_int = int(block_id)

    timestamp_int = int(timestamp_str)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    local_timestamp = int(time.time() * 1000)
    timestamp_diff = local_timestamp - timestamp_int
    hash = compute_truncated_sha256(decompressed_data)
    print(f"{timestamp}: diff: {timestamp_diff} ms block_id: {block_id} {hash}")

    if next_starting_id is not None and block_id_int >= next_starting_id:
        cumulative_hash = compute_truncated_sha256(cumulative_hash + hash)
        hash_counter += 1

        if hash_counter == 30:
            start_range_id = block_id_int - 29
            print(f"Cumulative Hash for range {start_range_id}-{block_id}: {cumulative_hash}")
            #await websocket.send(f"CP:{cumulative_hash}")
            cumulative_hash = ""
            hash_counter = 0

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
    pong_wait = True
    next_starting_id = None

    while pong_wait or next_starting_id is not None:
        message = await websocket.recv()

        if message == "Pong":
            if pong_wait:
                print("Received: Pong")
                if len([_async for _async in asyncio.all_tasks() if _async.get_name() == "hello_task"]) < 5:
                    continue
                else:
                    pong_wait = False
                    print("Server is ready for transmission!")
        else:
            decomp_data = zlib.decompress(message).decode('utf-8')
            if decomp_data.startswith("CP:"):
                _, range_info = decomp_data.split(":", 1)
                start_id, end_id = map(int, range_info.split('-'))
                print(f"Received checkpoint for range {start_id}-{end_id}")
                next_starting_id = end_id + 1
            else:    
                await process_data(websocket, message, next_starting_id)

# Sending "Hello" messages
async def send_hello_messages(websocket):
    hello_count = 0
    while hello_count < 5:
        await websocket.send("Hello")
        hello_count += 1
        await asyncio.sleep(1)

# Main coroutine
async def main():
    while True:
        try:
            async with websockets.connect('ws://xenblocks.io:6667') as websocket:
                print("Connected to the server!")
                reader_task = asyncio.create_task(websocket_reader(websocket), name="reader_task")
                hello_task = asyncio.create_task(send_hello_messages(websocket), name="hello_task")
                await asyncio.gather(reader_task, hello_task)
        except Exception as e:
            print(f"Connection error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

# Initialize the database
init_db()

# Run the main coroutine
asyncio.run(main())

