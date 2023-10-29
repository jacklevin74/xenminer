import asyncio
import websockets
import zlib
from datetime import datetime
import time
import hashlib
import sqlite3

DATABASE_NAME = 'blocks.db'
ready_flag = False

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

# Process the received data
async def process_data(message):
    decompressed_data = zlib.decompress(message).decode('utf-8')
    parts = decompressed_data.split('|')

    block_id, hash_to_verify, key, account, created_at, timestamp_str = parts
    timestamp_int = int(timestamp_str)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    local_timestamp = int(time.time() * 1000)
    timestamp_diff = local_timestamp - timestamp_int
    hash = compute_truncated_sha256(decompressed_data)
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

        if message == "Pong":
            pong_count += 1
            print("Received: Pong")
            if pong_count == 5:
                ready_flag = True
                print("Server is ready for transmission!")
        else:
            if ready_flag:
                await process_data(message)

# Sending "Hello" messages
async def send_hello_messages(websocket):
    while not ready_flag:
        await websocket.send("Hello")
        await asyncio.sleep(1)

# Main coroutine
async def main():
    global ready_flag
    while True:
        ready_flag = False  # Reset the ready flag each time before connecting
        pong_count = 0      # Reset the pong count as well
        try:
            async with websockets.connect('ws://xenblocks.io:6667') as websocket:
                print("Connected to the server!")
                reader_task = asyncio.create_task(websocket_reader(websocket))
                hello_task = asyncio.create_task(send_hello_messages(websocket))
                await asyncio.gather(reader_task, hello_task)
        except Exception as e:
            print(f"Connection error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


# Initialize the database
init_db()

# Run the main coroutine
asyncio.get_event_loop().run_until_complete(main())

