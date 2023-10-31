import asyncio
import websockets
import json
from collections import deque
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Function to verify the hash
def verify_argon2id_hash(hashed_data, key):
    ph = PasswordHasher()
    try:
        ph.verify(hashed_data, key)
        return True
    except VerifyMismatchError:
        return False

# Store the last 100 block IDs
recent_block_ids = deque(maxlen=100)

async def get_data():
    uri = "ws://186.233.186.56:8765"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    await websocket.send("request")
                    response = await websocket.recv()
                    data = json.loads(response)

                    for block in data:
                        block_id, argon2id_hash, key, *rest = block.split('|')

                        # Only verify if the block ID is not in recent_block_ids
                        if block_id not in recent_block_ids:
                            is_valid = verify_argon2id_hash(argon2id_hash, key)
                            if is_valid:
                                print(f"Verified Block ID: {block_id}")

                            # Add the block ID to the deque
                            recent_block_ids.append(block_id)

                    await asyncio.sleep(3)
        except websockets.ConnectionClosed:
            print("Connection lost, attempting to reconnect...")
            await asyncio.sleep(5)  # Wait for 5 seconds before reconnecting
        except Exception as e:
            print(f"Error occurred: {e}")
            await asyncio.sleep(5)  # Wait for 5 seconds before attempting to reconnect

# Run the coroutine
asyncio.run(get_data())

