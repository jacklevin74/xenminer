from flask import Flask, request, jsonify, render_template
from passlib.hash import argon2
import base64
from datetime import datetime
import time
import re
from web3 import Web3

app = Flask(__name__)

# Global variables to hold cached difficulty level and the time it was fetched
cached_difficulty = None
last_fetched_time = 0

from flask import Flask, render_template

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


# Function to get difficulty level
def get_difficulty(account=None):
    # fetch from consensus
    return 8


@app.route('/difficulty', methods=['GET'])
def difficulty(account=None):
    difficulty_level = get_difficulty(account)
    # Check if difficulty level exists
    if difficulty_level:
        return jsonify({"difficulty": difficulty_level}), 200
    else:
        return jsonify({"error": "Difficulty level not found."}), 404


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
    difficulty = get_difficulty()
    submitted_difficulty = int(re.search(r'm=(\d+)', hash_to_verify).group(1))
    strict_check = False

    if f'm={difficulty}' in hash_to_verify and is_xuni_present:
        strict_check = True

    if submitted_difficulty < int(difficulty): 

        print ("This Generates 401 for difficulty being too low", submitted_difficulty, int(difficulty))
        error_message = f"Hash does not contain 'm={difficulty}'. Your memory_cost setting in your miner will be autoadjusted."
        log_verification_failure(error_message, account)
        return jsonify({"message": error_message}), 401

    
    stored_targets = ['XEN1']  # Adjusted list to exclude 'XUNI' since we will search for it differently
    found = False

    for target in stored_targets:
        if target in hash_to_verify[-87:]:
            found = True
            print("Found Target:", target)

    # Search for XUNI followed by a number
    if re.search('XUNI[0-9]', hash_to_verify[-87:]) is not None:
        found = True
        print("Found Target: XUNI[0-9]")

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

    if is_verified: 
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        is_xen11_present = 'XEN1' in hash_to_verify[-87:]
        is_xuni_present = re.search('XUNI[0-9]', hash_to_verify[-87:]) is not None
        # disable XUNI
        # is_xuni_present = False

        # If XUNI is present and time is within 5 minutes of the hour, then insert to DB
        if is_xuni_present and is_within_five_minutes_of_hour():
            print("XUNI submitted and added to batch")

        elif is_xen11_present:  # no time restrictions for XEN11
            print("XEN11 hash added to batch")

        else:
            return jsonify({"message": "XUNI found outside of time window"}), 401


        return jsonify({"message": "Hash verified successfully and block saved."}), 200

    else:
        print ("Hash verification failed")
        return jsonify({"message": "Hash verification failed."}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
