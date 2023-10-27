import asyncio
import time
import websockets
import zlib
import sqlite3
import hashlib
from datetime import datetime

DB_PATH = 'blocks.db'
MAX_ERRORS = 100  # Maximum allowed errors before blocking a client

# Initialize global variables
connected_clients = {}  # Mapping client IP to WebSocket object
hello_count = {}        # Counting 'Hello' messages per client
message_queues = {}     # Mapping client IP to its asyncio queue
new_data_event = {}     # Event to notify new data availability
last_processed_id = 0   # Initialize the last processed ID
error_counters = {}     # Error counters for each client

async def print_error_counters_periodically():
    while True:
        print("Current Error Counters:", error_counters)
        await asyncio.sleep(10)

def compute_truncated_sha256(s):
    sha256_hash = hashlib.sha256(s.encode()).hexdigest()
    truncated_hash = sha256_hash[:5]
    return truncated_hash

def init_db():
    global last_processed_id
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(block_id) FROM blocks")
    row = cursor.fetchone()
    last_processed_id = row[0] if row and row[0] else 0
    conn.close()

# Handle WebSocket messages
async def server_handler(websocket, path):
    client_ip = websocket.remote_address[0]
    hello_count[client_ip] = 0
    message_queues[client_ip] = asyncio.Queue()
    new_data_event[client_ip] = asyncio.Event()
    connected_clients[client_ip] = websocket

    try:
        async for message in websocket:
            if message == "Hello":
                hello_count[client_ip] += 1
                await websocket.send("Pong")
                if hello_count[client_ip] == 5:
                    print(f"Client {client_ip} is ready for transmission!")
                    if error_counters.get(client_ip, 0) <= MAX_ERRORS:
                        asyncio.create_task(send_messages(client_ip))
                    else:
                        print(f"Client {client_ip} is blocked due to excessive errors.")
            else:
                print(f"Received unknown message from {client_ip}: {message}")
    except websockets.exceptions.ConnectionClosedOK:
        print(f"Connection closed cleanly with {client_ip}")
    except Exception as e:
        print(f"Error with client {client_ip}: {e}")
    finally:
        clean_up_client(client_ip)

def clean_up_client(client_ip):
    """ Function to clean up client data from various structures. """
    if client_ip in connected_clients:
        del connected_clients[client_ip]
    if client_ip in hello_count:
        del hello_count[client_ip]
    if client_ip in message_queues:
        del message_queues[client_ip]
    if client_ip in new_data_event:
        del new_data_event[client_ip]

# Send messages to clients
async def send_messages(client_ip):
    ws = connected_clients[client_ip]
    mq = message_queues[client_ip]

    while True:
        try:
            data = await mq.get()
            compressed_data = zlib.compress(data.encode('utf-8'))
            await ws.send(compressed_data)
            await asyncio.sleep(0.005)
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection closed with client {client_ip}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error with client {client_ip}: {e}")
            error_counters[client_ip] = error_counters.get(client_ip, 0) + 1
            if error_counters[client_ip] > MAX_ERRORS:
                print(f"Exceeded max errors for {client_ip}. Blocking communication.")
                break

async def query_data():
    global last_processed_id
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    while True:
        cursor.execute("SELECT block_id, hash_to_verify, key, account, created_at FROM blocks WHERE block_id > ?", (last_processed_id,))
        rows = cursor.fetchall()
        for row in rows:
            id, hash_to_verify, key, account, created_at = row
            timestamp = str(int(time.time() * 1000))
            whole_row = f"{id}|{hash_to_verify}|{key}|{account}|{created_at}|{timestamp}"
            hash = compute_truncated_sha256(whole_row)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} {hash}")

            for ip in connected_clients:
                await message_queues[ip].put(whole_row)

            last_processed_id = id
            print("Total Nodes Connected: ", len(connected_clients))

        if rows:
            for ip in connected_clients:
                new_data_event[ip].set()

        await asyncio.sleep(0.5)
    conn.close()

init_db()

async def start_server():
    # Start the WebSocket server
    async with websockets.serve(server_handler, "0.0.0.0", 6667, max_size=None, ping_interval=None) as server:
        print_task = asyncio.create_task(print_error_counters_periodically())
        query_task = asyncio.create_task(query_data())

        # Wait for all tasks to complete
        await asyncio.gather(print_task, query_task)

# Run the server
asyncio.run(start_server())
