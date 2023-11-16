from flask import Flask, request, jsonify
import secrets
import sqlite3
from ethereum.transactions import Transaction
from flask_cors import cross_origin
from ethapi.api import EthApi
from ethapi.utils import encode_block
from flask import abort
import rlp
from web3 import Web3
from flask_cors import CORS
import logging

eth_api = EthApi("sqlite:///blockchain.db")
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)


@app.route('/')
@cross_origin()
def hello_world():
    return 'ok'


# Function to fetch balance from SQLite database
def validate_transaction(tx):
    # Validate the transaction (this is a placeholder; your validation logic goes here)
    return True


def broadcast_transaction(tx):
    # Broadcast the transaction to the network (this is a placeholder; your broadcast logic goes here)
    # Returning a fake hash for demonstration
    return "0x" + tx.hash.hex()


def get_xuni_account_count(account_name):
    # Initialize database connection
    conn = sqlite3.connect('blocks.db')
    cursor = conn.cursor()

    try:
        # Run the SQL querye
        cursor.execute("SELECT COUNT(*) as n FROM xuni WHERE account = ?", (account_name,))
        data = cursor.fetchone()

        # Close database connection
        conn.close()

        # Check if data was found
        if data:
            return data[0]
        else:
            return -1  # can use -1 or another indicator to show that account was not found

    except sqlite3.Error as e:
        logging.error("Database error: %s", e)
        # Close database connection in case of an error
        conn.close()
        return -2  # can use -2 or another indicator to show that there was a database error


# Function to update the account balances in the super_blocks table
def transfer(from_account, to_account, value):
    try:
        conn_cache = sqlite3.connect("cache.db")
        cursor_cache = conn_cache.cursor()

        conn = sqlite3.connect("blocks.db")
        cursor = conn.cursor()
        # Check if the value is negative
        if value < 0:
            print("Transfer value cannot be negative.")
            return

        # Debug print
        print(f"Initiating transfer of {value} from {from_account} to {to_account}")

        # Add '0x' prefix if not present and convert accounts to lower case
        #from_account = from_account.lower() if from_account.startswith('0x') else '0x' + from_account.lower()
        #to_account = to_account.lower() if to_account.startswith('0x') else '0x' + to_account.lower()

        # Check balance of the sender account
        #cursor.execute("SELECT super_block_count FROM super_blocks WHERE LOWER(account) = ?", (from_account,))

        #row = cursor.fetchone()
        
        #if row:
        #    current_balance = row[0]
        #    if current_balance < value:
        #        print(f"Insufficient balance in account {from_account}. Transfer aborted.")
        #        return
        #else:
        #    print(f"Account {from_account} not found. Transfer aborted.")
        #    return

        # Deduct the value from the sender account
        #cursor.execute("""
        #    UPDATE super_blocks
        #    SET super_block_count = super_block_count - ?
        #    WHERE LOWER(account) = ?;
        #""", (value, from_account))

        #affected_rows = cursor.rowcount
        #print(f"Deducted {value} from {from_account}. Rows affected: {affected_rows}")

        # Add the value to the receiver account
        #cursor.execute("""
        #    UPDATE super_blocks
        #    SET super_block_count = super_block_count + ?
        #    WHERE LOWER(account) = ?;
        #""", (value, to_account))

        #if cursor.rowcount == 0:
            #print(f"Account {to_account} not found. Inserting a new row.")
            #cursor.execute("INSERT INTO super_blocks (account, super_block_count) VALUES (?, ?)", (to_account, value))

        # Commit and close
        #conn.commit()
        conn.close()

    except sqlite3.Error as e:
        logging.error("SQLite error occurred: %s", e)
        if conn:
            conn.rollback()
            logging.error("Transaction rolled back.")
        raise


def get_xblk_account_count(account):
    try:
        conn = sqlite3.connect("cache.db")
        cursor_cache = conn.cursor()

        cursor_cache.execute("SELECT super_blocks FROM cache_table WHERE LOWER(account) = LOWER(?)", (account,))
        logging.debug("Account: %s", account)
        row = cursor_cache.fetchone()
        return row[0] if row else 0
    except Exception as e:
        logging.error("Database error:", e)
        return 0
    finally:
        if conn:
            conn.close()


def get_balance_from_db(account):
    try:
        conn = sqlite3.connect("blocks.db")
        cursor = conn.cursor()

        conn_cache = sqlite3.connect("cache.db")
        cursor_cache = conn_cache.cursor()

        query = "SELECT super_block_count FROM super_blocks WHERE LOWER(account) = LOWER(?)"
        #print(f"Executing SQL query: {query} with account: {account}")  # Print the query and account to standard output
        #cursor.execute("SELECT super_block_count FROM super_blocks WHERE LOWER(account) = LOWER(?)", (account,))
        print("Account: ", account)
        cursor_cache.execute("SELECT total_blocks FROM cache_table WHERE LOWER(account) = ?", (account.lower(),))
        row = cursor_cache.fetchone()
        print("Balance for ", account, row[0] * 10 if row else 0)  # Modified this line to handle None
        return row[0] * 10 if row else 0
    except Exception as e:
        logging.error("Database error: %s", e)
        return 0
    finally:
        if conn:
            conn.close()
            conn_cache.close()


def rlp_encode(input_string):
    if len(input_string) == 1 and ord(input_string) < 0x80:
        return input_string
    elif len(input_string) <= 55:
        return chr(0x80 + len(input_string)) + input_string
    else:
        length_of_length = len(str(len(input_string)))
        return chr(0xb7 + length_of_length) + str(len(input_string)) + input_string


def handle_eth_call(data):
    logging.debug("In handle_eth_call function: %s", data)
    
    # Check if 'data' key exists
    if 'params' not in data or not isinstance(data['params'], list) or len(data['params']) == 0 or 'data' not in data['params'][0]:
        logging.error("Data missing: %s", data)
        response = "0x123456"
        return response

    # Extracting the relevant details from the data
    target_address = data['params'][0]['to'].lower()

    # Default names and symbols for different contracts
    contract_data = {
        "0x999999cf1046e68e36e1aa2e0e07105eddd00002": {"name": "XUNI Token", "symbol": "XUNI"},
        "0x999999cf1046e68e36e1aa2e0e07105eddd00001": {"name": "X.BLK Token", "symbol": "X.BLK"}
    }

    # If the contract address is not recognized
    if target_address not in contract_data:
        return {
            "id": data['id'],
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": "Unsupported contract address"
            }
        }

    function_data = data['params'][0]['data']
    function_signature = function_data[:10]
    address_queried = function_data[10:74]
    logging.debug("Function Signature: %s", function_signature)

    if function_signature == '0x313ce567':  # decimals function
        logging.debug("RETURN DECIMALS")
        decimals = 18
        response = '0x' + hex(decimals)[2:].zfill(64)

    elif function_signature == '0x06fdde03':  # name function
        token_name = contract_data[target_address]["name"]
        length_in_hex = hex(len(token_name))[2:].zfill(64)
        encoded_name = token_name.encode().hex().ljust(64, '0')
        response = '0x' + '0000000000000000000000000000000000000000000000000000000000000020' + length_in_hex + encoded_name
        logging.debug("RETURN NAME: %s", token_name)

    elif function_signature == '0x95d89b41':  # symbol function
        symbol = contract_data[target_address]["symbol"]
        length_in_hex = hex(len(symbol))[2:].zfill(64)
        encoded_symbol = symbol.encode().hex().ljust(64, '0')
        response = '0x' + '0000000000000000000000000000000000000000000000000000000000000020' + length_in_hex + encoded_symbol
        logging.debug("RETURN SYMBOL: %s", symbol)

    elif function_signature == '0x70a08231':  # balanceOf function
        address_queried = function_data[34:74].lower()

        # Depending on the target address, call the appropriate function to get the balance
        if target_address == "0x999999cf1046e68e36e1aa2e0e07105eddd00002":
            balance = get_xuni_account_count('0x' + address_queried) * 1000000000000000000
        elif target_address == "0x999999cf1046e68e36e1aa2e0e07105eddd00001":
            balance = get_xblk_account_count('0x' + address_queried) * 1000000000000000000
        else:
            # Handle unknown contract address or give a default balance
            balance = 0

        response = '0x' + hex(balance)[2:].zfill(64)
        logging.debug("RETURN BALANCE for: %s", address_queried)
        logging.debug("BALANCE is: %s", balance)

    else:
        response = {
            "id": data['id'],
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": "Invalid request"
            }
        }

    logging.debug(response)
    return response


@app.route('/', methods=['POST'])
@cross_origin()
def index():
    data = request.get_json()
    logging.info("Received data: %s", data)  # Print received queries to the screen
    if not data:
        abort(400, description="No data provided")

    if isinstance(data, list):
        # Handle batch request
        response = []
        for item in data:
            response.append(handle_rcp_call(item))
    else:
        response = handle_rcp_call(data)

    logging.debug("sending response: %s", response)
    return jsonify(response)


def handle_rcp_call(data):
    # Initialize the result variable
    result = None

    if data['jsonrpc'] != '2.0':
        return {'jsonrpc': '2.0', 'error': {'code': -32600, 'message': 'Invalid Request'}, 'id': None}

    if data['method'] == 'eth_blockNumber':
        result = hex(eth_api.block_number())

    elif data['method'] == 'eth_getBalance':
        account = data['params'][0]  # Convert account to lower case to ensure it matches
        balance_decimal = get_balance_from_db(account)  # Fetch balance from the database
        result = hex(int(balance_decimal * 10**18))  # Convert to Wei

    elif data['method'] == 'eth_estimateGas':
        # Simulate gas estimation here. This is a simplified example and
        # you would usually perform an actual gas estimation based on the
        # transaction details.
        result = '0x5208'  # 21000 in hex, a common gas cost for simple transfers
    
    elif data['method'] == 'eth_call':
        print("ETH CALL RECEIVED: ")
        print(data)
        result = handle_eth_call(data)
    
    elif data['method'] == 'eth_chainId':
        result = '0x18705'  # Mainnet

    elif data['method'] == 'eth_getCode':
        address = data['params'][0].lower()  # Convert the address to lower case
        block_number = data['params'][1]  # The block number or tag; you may or may not use this depending on your implementation

        # Example: Fetch the contract code from your database or some storage.
        # In this example, it's hardcoded.
        contract_code = {
            '0xdadf7ac7d0622dedd7e58b4d85d3784cc0c9d0e7': '0x606060405260...',
            '0x999999cf1046e68e36e1aa2e0e07105eddd00002': '0x00002',
            '0x999999cf1046e68e36e1aa2e0e07105eddd00001': '0x00001'
        }

        result = contract_code.get(address, '0x')  # return '0x' if the address is not a contract

    elif data['method'] == 'eth_getTransactionReceipt':
        tx_hash = data['params'][0]
        print("Entering eth_getTransactionReceipt")

        # Connect to SQLite database
        conn = sqlite3.connect('blocks.db')
        cursor = conn.cursor()

        # Query the transactions table
        cursor.execute("SELECT from_account, to_account, value FROM transactions WHERE tx_hash=?", (tx_hash,))
        row = cursor.fetchone()

        if row:
            from_account, to_account, value = row

            # Populate receipt with database values
            receipt = {
                'transactionHash': tx_hash,
                'transactionIndex': '0x1',
                'blockHash': '0x' + secrets.token_hex(32),
                'blockNumber': '0x6',
                'from': from_account,
                'to': to_account,
                'cumulativeGasUsed': '0xA',
                'gasUsed': '0xA',
                'contractAddress': None,
                'logs': [],
                'status': '0x1'
            }
            print("Sending: ", receipt)
            result = receipt

        else:
            print(f"No transaction found for hash {tx_hash}")

        conn.close()

    elif data['method'] == 'eth_getTransactionCount':
        address = data['params'][0].lower()
        block_number = data['params'][1]  # May or may not use this depending on your needs

        # Fetch the nonce from your database using the get_nonce function
        nonce = get_nonce(address)
        print("Returning nonce: ", nonce)
        result = hex(nonce)
        
        print("Returning nonce hex: ", result)

    elif data['method'] == '1eth_getTransactionCount':
        address = data['params'][0].lower()
        block_number = data['params'][1]  # May or may not use this depending on your needs

        # Fetch the nonce from your database or however you're storing nonces
        # For demonstration, we'll just use a hardcoded example
        nonces = {
            "0xc855fd5aa2829799dde83b43ac33651e46f610bb": 10
        }
        result = hex(nonces.get(address, 0))
        print("Returning nonce hex: ", result)

    elif data['method'] == 'Xeth_getBlockByNumber':
        result = {
            'number': '0x5BAD55',
            'hash': '0x1234567890abcdef',
            'transactions': []
        }

    elif data['method'] == 'eth_getBlockByNumber':
        if "params" not in data or len(data["params"]) < 2:
            return {'jsonrpc': '2.0', 'error': {'code': -32602, 'message': 'Invalid params'}, 'id': data.get('id', None)}

        block_number = data['params'][0]
        full_tx = data['params'][1]
        res = eth_api.block_by_number(block_number, full_tx=full_tx)
        if res:
            result = encode_block(res)
    
    elif data['method'] == 'net_version':
        result = '1'  # Mainnet

    elif data['method'] == 'eth_getBlockByHash':
        if "params" not in data or len(data["params"]) < 2:
            return {'jsonrpc': '2.0', 'error': {'code': -32602, 'message': 'Invalid params'}, 'id': data.get('id', None)}

        block_hash = data['params'][0]
        full_tx = data['params'][1]
        res = eth_api.block_by_hash(block_hash, full_tx)
        if res:
            result = encode_block(res)

    elif data['method'] == 'eth_gasPrice':
        result = '0x3B9ACA00'  # Example gas price, 1 Gwei in hexadecimal

    elif data['method'] == 'eth_sendRawTransaction':
        print(data)
        raw_tx = data['params'][0]
        print("Raw TX: ", raw_tx)
        try:
            handle_raw_transaction(raw_tx)
            tx_hash = get_transaction_hash(raw_tx)
            print("Returning results from eth_sendRawTransaction ", tx_hash)
            result = tx_hash
        except Exception as e:
            return {'jsonrpc': '2.0', 'error': {'code': -32000, 'message': f'Exception: {e}'}, 'id': data['id']}

    else:
        return {'jsonrpc': '2.0', 'error': {'code': -32601, 'message': 'Method not found'}, 'id': data.get('id', None)}

    response = {
        'jsonrpc': '2.0',
        'id': data.get('id', None),
        'result': result
    }

    return response


def get_recovered_address(raw_tx):
    w3 = Web3()
    try:
        # Recover the sender's address from the raw transaction
        recovered_address = w3.eth.account.recover_transaction(raw_tx)
        return recovered_address
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_nonce(from_account):
    conn = sqlite3.connect('blocks.db')
    cursor = conn.cursor()
    # Count the number of transactions for the from_account
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE from_account = ?", (from_account,))
    count = cursor.fetchone()[0]
    
    # If there's at least one transaction, increment the count to get the next nonce.
    # If there are no transactions, the count will be 0 and that's the nonce you'll use.
    return count + 1 if count else 0


def get_transaction_hash(raw_tx):
    return Web3.keccak(hexstr=raw_tx).hex()


def handle_raw_transaction(raw_tx):
    # Decode the raw transaction
    conn = sqlite3.connect('blocks.db')
    c = conn.cursor()

    try:
        # Decoding the hex-encoded transaction
        decoded_tx = rlp.decode(bytes.fromhex(raw_tx[2:]), Transaction)

        #tx_hash = "0x" + decoded_tx.hash().hex()
        #tx_hash = broadcast_transaction(raw_tx)

        tx_hash = get_transaction_hash(raw_tx) 
        print("TX hash: ", tx_hash)

        # Extract details
        from_account = decoded_tx.sender  # Or use another way to get sender address
        to_account = decoded_tx.to
        value = decoded_tx.value
        gas = decoded_tx.startgas
        gas_price = decoded_tx.gasprice
        nonce = decoded_tx.nonce
        input_data = decoded_tx.data
        # Validate from address
        validated_from = get_recovered_address(raw_tx)

        from_account = validated_from

        # Get the nonce value
        nonce = get_nonce(from_account)

        # Insert the new record into the transactions table
        to_acc = "0x" + to_account.hex()
        print("Inserting: ", tx_hash, raw_tx, from_account, to_acc, value, nonce)
        c.execute('''INSERT INTO transactions (tx_hash, raw_tx, from_account, to_account, value, nonce)
                     VALUES (?, ?, ?, ?, ?, ?)''', (tx_hash, raw_tx, from_account.lower(), to_acc.lower(), value, nonce))
        
        # Commit the transaction
        conn.commit()
        # Close the connection
        conn.close()

        transfer (validated_from, to_account.hex(), value/1e18)

        # Print the decoded transaction details
        print(f"From Account: {validated_from}")
        print(f"To Account: {to_account.hex() if to_account else 'Contract Creation'}")
        print(f"Value: {value} wei")
        print(f"Gas: {gas}")
        print(f"Gas Price: {gas_price} wei")
        print(f"Nonce: {nonce}")
        print(f"Input Data: {input_data.hex()}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
