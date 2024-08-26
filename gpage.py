from flask import Flask, request, jsonify, render_template, redirect
from requests.exceptions import RequestException
from passlib.hash import argon2
import sqlite3, base64
from datetime import datetime
import requests
import time
import re
from web3 import Web3

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

from datetime import datetime
def is_within_five_minutes_of_hour():
    timestamp = datetime.now()
    minutes = timestamp.minute
    print ("My minutes ", minutes)
    return 0 <= minutes < 5 or 55 <= minutes < 60

# Specify the file path where you want to save the messages
log_file_path = './error_log_filr.log'

def log_verification_failure(message, account):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file_path = "your_log_file.log"
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{current_time} - Issued 401: {account}. Message: {message}\n")


def read_difficulty_level(file_path):
    try:
        with open(file_path, "r") as file:
            # Read the file and strip any leading/trailing whitespace
            difficulty_level = file.read().strip()
            return difficulty_level
    except FileNotFoundError:
        print("File not found.")
        return 100000
    except Exception as e:
        print(f"An error occurred: {e}")
        return 100000


# Function to get difficulty level
def get_difficulty(account=None):

    file_path = "/home/ubuntu/mining/diff2.chain"
    try:
        new_difficulty_level = int(read_difficulty_level(file_path))
        #print ("Read from file: ", new_difficulty_level)
    except (TypeError, ValueError):
        new_difficulty_level = 100000

    return str(new_difficulty_level)

# Function to get difficulty level
def get_difficulty2(account=None):
    global cached_difficulty, last_fetched_time  # Declare as global to modify them

    # Check if it has been more than 60 seconds since the last fetch
    current_time = time.time()
    if current_time - last_fetched_time < 10:
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


@app.route('/get_xuni_counts', methods=['GET'])
def get_account_counts():
    # Initialize database connection
    conn = sqlite3.connect('blocks.db')
    cursor = conn.cursor()

    try:
        # Run the SQL query
        cursor.execute("SELECT account, COUNT(*) as n FROM xuni GROUP BY account ORDER BY n;")
        data = cursor.fetchall()

        # Close database connection
        conn.close()

        # Prepare the result in JSON format
        result = [{"account": account, "count": n} for account, n in data]

        return jsonify(result)

    except sqlite3.Error as e:
        print("Database error:", e)
        # Close database connection in case of an error
        conn.close()
        return jsonify({"error": "Database error"}), 500


@app.route('/blockrate_per_day', methods=['GET'])
def blockrate_per_day():
    try:
        with sqlite3.connect('blocks.db') as conn:
            c = conn.cursor()

            c.execute('''
            SELECT account, num_blocks 
            FROM AccountBlockCounts 
            ORDER BY num_blocks DESC 
            LIMIT 1000
            ''')

            rows = c.fetchall()

            # Convert rows into a list of dictionaries for JSON representation
            users_list = [{"account": row[0], "num_blocks": row[1]} for row in rows]

            return jsonify(users_list), 200

    except Exception as e:
        return jsonify({"error": "An error occurred: " + str(e)}), 500

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    return redirect("https://explorer.xenblocks.io")

@app.route('/get_balance/<account>', methods=['GET'])
def get_balance(account):
    conn = None
    try:
        conn = sqlite3.connect('cache.db', timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT total_blocks FROM cache_table WHERE LOWER(account) = LOWER(?)", (account,))
        row = cursor.fetchone()
        if row:
            balance = row[0] * 10
            return jsonify({'account': account, 'balance': balance})
        else:
            return jsonify({'error': 'No record found for the provided account'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route('/get_super_blocks/<account>', methods=['GET'])
def get_super_blocks(account):
    conn = None
    try:
        conn = sqlite3.connect('cache.db', timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT super_blocks FROM cache_table WHERE LOWER(account) = LOWER(?)", (account,))
        row = cursor.fetchone()
        if row:
            return jsonify({'account': account, 'super_blocks': row[0]})
        else:
            return jsonify({'error': 'No record found for the provided account'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            conn.close()


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
                 FROM (SELECT * FROM account_attempts ORDER BY timestamp DESC LIMIT 50000)''')

    # Rest of your code
    result = c.fetchone()
    conn.close()

    total_attempts, total_time = result
    total_attempts_per_second = total_attempts / (total_time if total_time != 0 else 1)

    return render_template('hash_rate.html', total_attempts_per_second=total_attempts_per_second)

# Temporary storage for batch insertion
account_attempts_batch = []
blocks_batch = []
batch_size = 1

def is_valid_sha256(s):
    """Check if s is a valid SHA-256 hash."""
    return re.match(r'^[a-fA-F0-9]{64}$', s) is not None

def is_hexadecimal(s):
    """Check if s is a hexadecimal string."""
    return re.match(r'^[a-fA-F0-9]*$', s) is not None

def check_fourth_element(string):
    pattern = re.compile(r'(?:[^$]*\$){3}WEVOMTAwODIwMjJYRU4\$')
    match = pattern.search(string)
    return bool(match)

def is_valid_hash(h):
    """Ensure the input is a hexadecimal hash of the expected length."""
    return bool(re.match("^[a-fA-F0-9]{64}$", h))

def restore_eip55_address(lowercase_address: str) -> str:
    # Restore the address using EIP-55 checksum
    try:
        checksum_address = Web3.to_checksum_address(lowercase_address)
        print ("Checksummed address is: ", checksum_address)
        return True
    except ValueError as e:
        # Handle the error in case the address is not a valid Ethereum address
        print(f"An error occurred: {e}")
        return False


def check_salt_format_and_ethereum_address(hash_to_verify: str) -> bool:
    # Regular expressions for the expected patterns
    pattern1 = re.compile(r'WEVOMTAwODIwMjJYRU4')
    pattern2 = re.compile(r'^[A-Za-z0-9+/]{27}$')

    # Extract the salt part from the hash_to_verify
    parts = hash_to_verify.split("$")
    if len(parts) != 6:
        return False
    salt = parts[4]

    # Check if the salt matches the first pattern
    if pattern1.search(salt):
        print ("Matched old salt, continue")
        return True
    else:
        print("Old Salt False")

    # Check if the salt matches the second pattern and is base64
    if pattern2.fullmatch(salt):
        print ("In Pattern2 match")
        try:
            # The proper base64 string should have a length that is a multiple of 4.
            # We need to add padding if necessary.
            missing_padding = len(salt) % 4
            if missing_padding:
                salt += '=' * (4 - missing_padding)

            # Decode the base64 string
            decoded_bytes = base64.b64decode(salt)
            decoded_str = decoded_bytes.hex()
            print("Decoded salt: ", decoded_str)

            # Check if the decoded string is a valid hexadecimal and of a specific length
            if re.fullmatch(r'[0-9a-fA-F]{40}', decoded_str):  # Ethereum addresses are 40 hex characters long
                # Construct potential Ethereum address
                potential_eth_address = '0x' + decoded_str
                print("Address matched: ", potential_eth_address)

                # Validate Ethereum address checksum
                if restore_eip55_address(potential_eth_address):
                    print("Checksum of address is valid")
                    return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    return False


@app.route('/get_block', methods=['GET'])
def get_block():
    key = request.args.get('key')
    if not key:
        return jsonify({"error": "Please provide a key"}), 400

    if not is_valid_hash(key):
        return jsonify({"error": "Invalid key provided"}), 400

    conn = sqlite3.connect('blocks.db')
    cursor = conn.cursor()

    # Use a parameterized query to prevent SQL injection
    cursor.execute("SELECT * FROM blocks WHERE key=?", (key,))
    data = cursor.fetchone()

    if data is None:
        # No record was found in the 'blocks' table, try 'xuni' table
        cursor.execute("SELECT * FROM xuni WHERE key=?", (key,))
        data = cursor.fetchone()

        if data is None:
            # Record not found in either table
            return jsonify({"error": "Data not found for provided key"}), 404

    # Column names for both 'blocks' and 'xuni' tables
    columns = ['block_id', 'hash_to_verify', 'key', 'account', 'created_at']

    # Convert the tuple data to a dictionary
    data_dict = dict(zip(columns, data))

    conn.close()

    return jsonify(data_dict), 200

@app.route('/verify', methods=['POST'])
def verify_hash():
    global account_attempts_batch, blocks_batch
    data = request.json
    worker_id = data.get('worker_id')

    if not (isinstance(worker_id, str) and len(worker_id) <= 3):
        worker_id = None  # Set worker_id to None if it's not a valid string of 3 characters or less

    hash_to_verify = data.get('hash_to_verify')
    hash_to_verify = hash_to_verify if (hash_to_verify and len(hash_to_verify) <= 150) else None
    is_xuni_present = re.search('XUNI[0-9]', hash_to_verify[-87:]) is not None
    key = data.get('key')
    key = key if (key and len(key) <= 128) else None
    account = data.get('account')

    if account is not None:
        account = str(account).lower().replace("'", "").replace('"', '')
        account = account if len(account) <= 43 else None

    attempts = data.get('attempts')
    difficulty = 0

    # Check if key is a hexadecimal string
    if not is_hexadecimal(key):
        return jsonify({"error": "Invalid key format"}), 400

    #if not check_fourth_element(hash_to_verify):
    if not check_salt_format_and_ethereum_address(hash_to_verify):
        return jsonify({"error": "Invalid salt format"}), 400

    # Check for missing data
    if not hash_to_verify or not key or not account:
        return jsonify({"error": "Missing hash_to_verify, key, or account"}), 400

    # Get difficulty level from the database
    old_difficulty = difficulty;
    difficulty = get_difficulty()
    submitted_difficulty = int(re.search(r'm=(\d+)', hash_to_verify).group(1))
    strict_check = False

    if f'm={difficulty}' in hash_to_verify and is_xuni_present:
        strict_check = True

    #if f'm={difficulty}' not in hash_to_verify:
    #    print ("Compare diff ", submitted_difficulty, int(difficulty))
    if submitted_difficulty < int(difficulty):
    #if abs(submitted_difficulty - int(difficulty)) > 50:

        print ("This Generates 401 for difficulty being too low", submitted_difficulty, int(difficulty))
        error_message = f"Hash does not contain 'm={difficulty}'. Your memory_cost setting in your miner will be autoadjusted."
        log_verification_failure(error_message, account)
        return jsonify({"message": error_message}), 401


    stored_targets = ['XEN11']  # Adjusted list to exclude 'XUNI' since we will search for it differently
    found = False

    for target in stored_targets:
        if target in hash_to_verify[-87:]:
            found = True
            print("Found Target:", target)

    # Search for XUNI followed by a number
    if re.search('XUNI[0-9]', hash_to_verify[-87:]) is not None:
        found = True
        print("Found Target: XUNI[0-9]")
        if not is_within_five_minutes_of_hour():
            error_message = f"XUNI Submitted outside of proper time frame."
            return jsonify({"message": error_message}), 401

    if not found:
        print (hash_to_verify)
        error_message = f"Hash does not contain any of the valid targets {stored_targets} in the last 87 characters. Adjust target_substr in your miner."
        log_verification_failure(error_message, account)
        print (error_message, hash_to_verify[-87:])
        return jsonify({"message": error_message}), 401

    if len(hash_to_verify) > 150:
        error_message = "Length of hash_to_verify should not be greater than 150 characters."
        print (error_message)
        log_verification_failure(error_message, account)
        return jsonify({"message": error_message}), 401

    try:
        is_verified = argon2.verify(key, hash_to_verify)
    except Exception as e:
        print(f"An error occurred: {e}")
        is_verified = False

    #if argon2.verify(key, hash_to_verify):
    if is_verified:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        is_xen11_present = 'XEN11' in hash_to_verify[-87:]
        is_xuni_present = re.search('XUNI[0-9]', hash_to_verify[-87:]) is not None
        # disable XUNI
        # is_xuni_present = False

        conn = sqlite3.connect('blocks.db')
        conn.execute('PRAGMA journal_mode = wal')
        c = conn.cursor()
        try:
            # If XUNI is present and time is within 5 minutes of the hour, then insert to DB
            if is_xuni_present and is_within_five_minutes_of_hour():
                print("XUNI submitted and added to batch")
                #c.execute('''INSERT INTO xuni (hash_to_verify, key, account)
                 #     VALUES (?, ?, ?)''', (hash_to_verify, key, account))

                #send_post_request(hash_to_verify, key, account, "1")
                insert_query = '''INSERT INTO xuni (hash_to_verify, key, account) VALUES (?, ?, ?)'''
                data_tuple = (hash_to_verify, key, account)
                success = insert_with_retry(c, insert_query, data_tuple)
                if not success:
                    print("Could not insert into the database after multiple retries.")

            elif is_xen11_present:  # no time restrictions for XEN11
                print("XEN11 hash added to batch")
                #c.execute('''INSERT INTO blocks (hash_to_verify, key, account)
                 #   VALUES (?, ?, ?)''', (hash_to_verify, key, account))


                #send to listener
                #send_post_request(hash_to_verify, key, account, "0")

                insert_query = '''INSERT INTO blocks (hash_to_verify, key, account) VALUES (?, ?, ?)'''
                data_tuple = (hash_to_verify, key, account)
                success = insert_with_retry(c, insert_query, data_tuple)
                if not success:
                    print("Could not insert into the database after multiple retries.")

            else:
                return jsonify({"message": "XUNI found outside of time window"}), 401

            #c.execute('''INSERT OR IGNORE INTO account_attempts (account, timestamp, attempts)
             #   VALUES (?, ?, ?)''', (account, timestamp, attempts))
            print("This Generates 200 for difficulty being good", submitted_difficulty, int(difficulty))
            print("Inserting hash into db: ", hash_to_verify)


            conn.commit()

        except sqlite3.IntegrityError as e:
            error_message = e.args[0] if e.args else "Unknown IntegrityError"
            print(f"Error: {error_message} ", hash_to_verify, key, account)
            return jsonify({"message": f"Block already exists, continue"}), 400

        finally:
            conn.close()

        return jsonify({"message": "Hash verified successfully and block saved."}), 200

    else:
        print ("Hash verification failed")
        return jsonify({"message": "Hash verification failed."}), 401


def send_post_request(hash_to_verify, key, account, kind):
    url = "http://localhost:9997/process_hash"  # Replace with your specific endpoint
    data = {
        "hash_to_verify": hash_to_verify,
        "key": key,
        "type": kind,
        "account": account
    }
    print ("ADDING REQUEST TO SEQUENCER ")
    print(data)

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Data sent successfully. Response:", response.status_code)
        else:
            print("Failed to send data. Status code:", response.status_code)
    except RequestException as e:
        print("An error occurred:", e)


def insert_with_retry(cursor, query, data, max_retries=10, sleep_interval=3):
    attempt = 0
    while attempt < max_retries:
        try:
            cursor.execute(query, data)
            return True  # Success
        except sqlite3.OperationalError as e:
            if str(e) == 'database is locked':
                print(f"Database is locked, retrying {attempt + 1}/{max_retries}")
                time.sleep(sleep_interval)  # Wait for the database to be unlocked
                attempt += 1
            else:
                raise  # Other operational errors are not handled here
    return False  # Max retries reached, could not insert


@app.route('/validate', methods=['POST'])
def store_consensus():
    data = request.json
    total_count = data.get('total_count')
    my_ethereum_address = data.get('my_ethereum_address')
    last_block_id = data.get('last_block_id')
    last_block_hash = data.get('last_block_hash')

    try:
        conn = sqlite3.connect('blocks.db')
        c = conn.cursor()
        c.execute('''INSERT INTO consensus (total_count, my_ethereum_address, last_block_id, last_block_hash)
                     VALUES (?, ?, ?, ?)''',
                     (total_count, my_ethereum_address, last_block_id, last_block_hash))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/top_daily_block_miners', methods=['GET'])
def get_top_blocks():
    conn = sqlite3.connect('blocks.db')  # Assuming your database file is named blocks.db
    cursor = conn.cursor()

    query = '''
        SELECT * FROM AccountBlockCounts
        ORDER BY num_blocks DESC
        LIMIT 500
    '''

    cursor.execute(query)
    rows = cursor.fetchall()

    # Close the connection
    conn.close()

    # Convert the rows to a JSON response
    result = [{"account": row[0], "num_blocks": row[1]} for row in rows]
    return jsonify(result)


@app.route('/latest_blockrate', methods=['GET'])
def get_latest_blockrate():
    try:
        # Connect to the difficulty database
        conn = sqlite3.connect('difficulty.db')
        cursor = conn.cursor()

        # Query the latest blockrate using "ORDER BY id DESC LIMIT 1" for better performance
        cursor.execute("SELECT id, date, rate FROM blockrate ORDER BY id DESC LIMIT 1")
        record = cursor.fetchone()

        if record is None:
            return jsonify({"error": "No blockrate data found"}), 404

        # Close the database connection
        conn.close()

        # Prepare the result in JSON format
        result = {
            "id": record[0],
            "date": record[1],
            "rate": record[2]
        }

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



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
