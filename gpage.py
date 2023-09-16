from flask import Flask, request, jsonify, render_template
from passlib.hash import argon2
import sqlite3
from datetime import datetime
import time
import re

app = Flask(__name__)

# Global variables to hold cached difficulty level and the time it was fetched
cached_difficulty = None
last_fetched_time = 0

def create_database():
    conn = sqlite3.connect('blocks.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS blocks (block_id INTEGER PRIMARY KEY AUTOINCREMENT, hash_to_verify TEXT, key TEXT UNIQUE, account TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS account_performance (account TEXT PRIMARY KEY, hashes_per_second REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS super_blocks (account TEXT, super_block_count INTEGER)')
    conn.commit()
    conn.close()

from flask import Flask, render_template
import sqlite3

app = Flask(__name__)


# Initialize cache dictionary and last fetched time
difficulty_cache = {}
last_fetched_time = {}

# Function to get difficulty level
def get_difficulty(account=None):
    global difficulty_cache, last_fetched_time  # Declare as global to modify them
    
    # Generate a cache key based on the account
    cache_key = account if account else "default"

    # Get the current time
    current_time = time.time()
    
    # Check if the cache has not expired (assuming 60 seconds as the cache duration)
    if cache_key in last_fetched_time and (current_time - last_fetched_time[cache_key] < 60):
        return difficulty_cache[cache_key]
    
    # Otherwise, fetch from database
    conn = sqlite3.connect('difficulty.db')  # Replace with your actual database name
    cursor = conn.cursor()

    if account:
        cursor.execute('SELECT difficulty FROM difficulty_table WHERE account = ? LIMIT 1;', (account,))
        difficulty_level = cursor.fetchone()
        if difficulty_level is None:
            cursor.execute('SELECT level FROM difficulty LIMIT 1;')
            difficulty_level = cursor.fetchone()
    else:
        cursor.execute('SELECT level FROM difficulty LIMIT 1;')
        difficulty_level = cursor.fetchone()

    conn.close()

    # Update the cache and the last fetched time
    if difficulty_level:
        difficulty_cache[cache_key] = str(difficulty_level[0])
    else:
        difficulty_cache[cache_key] = '8'  # or some default value

    last_fetched_time[cache_key] = current_time

    return difficulty_cache[cache_key]

# Function to get difficulty level
def get_difficulty2(account=None):
    global cached_difficulty, last_fetched_time  # Declare as global to modify them
    
    # Check if it has been more than 60 seconds since the last fetch
    current_time = time.time()
    if current_time - last_fetched_time < 60:
        return cached_difficulty

    # Connect to SQLite database
    conn = sqlite3.connect('blocks.db', timeout=10)  # Replace with your actual database name
    cursor = conn.cursor()

    # Execute SQL query to fetch difficulty level
    cursor.execute('SELECT level FROM difficulty LIMIT 1;')
    difficulty_level = cursor.fetchone()

    # Close connection
    conn.close()

    # Update last fetched time
    last_fetched_time = current_time

    # Update cached difficulty and return
    if difficulty_level:
        cached_difficulty = str(difficulty_level[0])
        return cached_difficulty
    else:
        cached_difficulty = '8'  # Return '8' or some default if no difficulty level is found
        return cached_difficulty

@app.route('/difficulty', methods=['GET'])
@app.route('/difficulty/<account>', methods=['GET'])
def difficulty(account=None):
    difficulty_level = get_difficulty(account)
    # Check if difficulty level exists
    if difficulty_level:
        return jsonify({"difficulty": difficulty_level}), 200
    else:
        return jsonify({"error": "Difficulty level not found."}), 404


@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    # Connect to the cache database
    cache_conn = sqlite3.connect('cache.db',timeout=10)
    cache_c = cache_conn.cursor()

    # Read from the cache table for leaderboard data
    cache_c.execute("SELECT * FROM cache_table ORDER BY total_blocks DESC LIMIT 500")
    results = cache_c.fetchall()
    cache_conn.close()

    # You could still calculate global statistics if needed from the original database
    conn = sqlite3.connect('blocks.db', timeout=10)
    c = conn.cursor()
    c.execute('''SELECT SUM(attempts) as total_attempts,
                 strftime('%s', MAX(timestamp)) - strftime('%s', MIN(timestamp)) as total_time
                 FROM (SELECT * FROM account_attempts ORDER BY timestamp DESC LIMIT 5000)''')
    result = c.fetchone()
    total_attempts, total_time = result
    total_attempts_per_second = total_attempts / (total_time if total_time != 0 else 1)
    conn.close()

    leaderboard = [(rank + 1, account, total_blocks, round(hashes_per_second, 2), super_blocks)
                   for rank, (account, total_blocks, hashes_per_second, super_blocks) in enumerate(results)]

    return render_template('leaderboard4.html', leaderboard=leaderboard,
                           total_attempts_per_second=int(round(total_attempts_per_second, 2) / 1000))


@app.route('/total_blocks', methods=['GET'])
def total_blocks():
    conn = sqlite3.connect('blocks.db')
    c = conn.cursor()

    # Query to get the latest block_id from the `blocks` table
    c.execute('SELECT block_id FROM blocks ORDER BY block_id DESC LIMIT 1')

    result = c.fetchone()
    last_block_id = result[0] if result else None

    conn.close()

    return jsonify({'total_blocks_top100': last_block_id}) 



@app.route('/hash_rate', methods=['GET'])
def hash_rate():
    conn = sqlite3.connect('blocks.db')
    c = conn.cursor()

    # Get the sum of all attempts and the time range for the last 10,000 records
    c.execute('''SELECT SUM(attempts) as total_attempts,
                 strftime('%s', MAX(timestamp)) - strftime('%s', MIN(timestamp)) as total_time
                 FROM (SELECT * FROM account_attempts ORDER BY timestamp DESC LIMIT 5000)''')

    # Rest of your code
    result = c.fetchone()
    conn.close()

    total_attempts, total_time = result
    total_attempts_per_second = total_attempts / (total_time if total_time != 0 else 1)

    return render_template('hash_rate.html', total_attempts_per_second=total_attempts_per_second)

# Temporary storage for batch insertion
account_attempts_batch = []
blocks_batch = []
batch_size = 10

@app.route('/verify', methods=['POST'])
def verify_hash():
    global account_attempts_batch, blocks_batch
    data = request.json
    hash_to_verify = data.get('hash_to_verify')
    key = data.get('key')
    account = data.get('account')
    attempts = data.get('attempts')

    # Check for missing data
    if not hash_to_verify or not key or not account:
        return jsonify({"error": "Missing hash_to_verify, key, or account"}), 400

    # Get difficulty level from the database
    difficulty = get_difficulty()
    if f'm={difficulty}' not in hash_to_verify:
        return jsonify({"message": f"Hash does not contain 'm={difficulty}'. Your memory_cost setting in your miner will be autoadjusted."}), 401

    if 'XEN11' not in hash_to_verify[-87:]:
        return jsonify({"message": "Hash does not contain 'XEN11' in the last 87 characters. Adjust target_substr in your miner."}), 401

    if len(hash_to_verify) > 136:
        return jsonify({"message": "Length of hash_to_verify should not be greater than 136 characters."}), 401

    if argon2.verify(key, hash_to_verify):
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            account_attempts_batch.append((account, timestamp, attempts))
            blocks_batch.append((hash_to_verify, key, account))

            # Check if batch size is reached
            if len(account_attempts_batch) >= batch_size:
                conn = sqlite3.connect('blocks.db')
                c = conn.cursor()

                # Batch insert for account_attempts
                c.executemany('''INSERT INTO account_attempts (account, timestamp, attempts)
                                VALUES (?, ?, ?)''', account_attempts_batch)

                # Batch insert for blocks
                c.executemany('''INSERT INTO blocks (hash_to_verify, key, account)
                                VALUES (?, ?, ?)''', blocks_batch)

                conn.commit()
                conn.close()

                # Clear the batches
                account_attempts_batch.clear()
                blocks_batch.clear()

                return jsonify({"message": "Hash verified successfully and block saved."}), 200

            else:
                return jsonify({"message": "Hash verified successfully, waiting for batch commit."}), 200

        except sqlite3.IntegrityError:
            return jsonify({"message": "Duplicate key rejected."}), 400

        finally:
            if 'conn' in locals():
                conn.close()

    else:
        return jsonify({"message": "Hash verification failed."}), 401

@app.route('/total_blocks2', methods=['GET'])
def total_blocks2():
    account = request.args.get('account')
    if not account:
        return jsonify({"error": "Missing account"}), 400

    conn = sqlite3.connect('blocks.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(block_id) FROM blocks WHERE account = ?', (account,))
    result = c.fetchone()
    conn.close()

    return jsonify({"total_blocks": result[0]}), 200
