from flask import Flask, request, jsonify
import secrets
import sqlite3
from ethereum.transactions import Transaction
from ethereum.utils import decode_hex
import rlp
from web3 import Web3

app = Flask(__name__)

# Initialize eth_blockNumber
current_block_number = 0x2234

# Function to fetch balance from SQLite database

def validate_transaction(tx):
    # Validate the transaction (this is a placeholder; your validation logic goes here)
    return True

def broadcast_transaction(tx):
    # Broadcast the transaction to the network (this is a placeholder; your broadcast logic goes here)
    # Returning a fake hash for demonstration
    return "0x" + tx.hash.hex()

# Function to update the account balances in the super_blocks table

def transfer(from_account, to_account, value):
    try:
        conn = sqlite3.connect("blocks.db")
        cursor = conn.cursor()
        # Check if the value is negative
        if value < 0:
            print("Transfer value cannot be negative.")
            return

        # Debug print
        print(f"Initiating transfer of {value} from {from_account} to {to_account}")

        # Add '0x' prefix if not present and convert accounts to lower case
        from_account = from_account.lower() if from_account.startswith('0x') else '0x' + from_account.lower()
        to_account = to_account.lower() if to_account.startswith('0x') else '0x' + to_account.lower()

        # Check balance of the sender account
        cursor.execute("SELECT super_block_count FROM super_blocks WHERE LOWER(account) = ?", (from_account,))
        row = cursor.fetchone()
        
        if row:
            current_balance = row[0]
            if current_balance < value:
                print(f"Insufficient balance in account {from_account}. Transfer aborted.")
                return
        else:
            print(f"Account {from_account} not found. Transfer aborted.")
            return

        # Deduct the value from the sender account
        cursor.execute("""
            UPDATE super_blocks
            SET super_block_count = super_block_count - ?
            WHERE LOWER(account) = ?;
        """, (value, from_account))

        affected_rows = cursor.rowcount
        print(f"Deducted {value} from {from_account}. Rows affected: {affected_rows}")

        # Add the value to the receiver account
        cursor.execute("""
            UPDATE super_blocks
            SET super_block_count = super_block_count + ?
            WHERE LOWER(account) = ?;
        """, (value, to_account))

        if cursor.rowcount == 0:
            print(f"Account {to_account} not found. Inserting a new row.")
            cursor.execute("INSERT INTO super_blocks (account, super_block_count) VALUES (?, ?)", (to_account, value))

        # Commit and close
        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        print("SQLite error occurred:", e)
        if conn:
            conn.rollback()
            print("Transaction rolled back.")
        raise




def get_balance_from_db(account):
    try:
        conn = sqlite3.connect("blocks.db")
        cursor = conn.cursor()
        query = "SELECT super_block_count FROM super_blocks WHERE LOWER(account) = LOWER(?)"
        #print(f"Executing SQL query: {query} with account: {account}")  # Print the query and account to standard output
        cursor.execute("SELECT super_block_count FROM super_blocks WHERE LOWER(account) = LOWER(?)", (account,))
        row = cursor.fetchone()
        return row[0] if row else 0
    except Exception as e:
        print("Database error:", e)
        return 0
    finally:
        if conn:
            conn.close()

@app.route('/', methods=['POST'])
def index():
    data = request.get_json()
    global current_block_number
    print("Received data:", data)  # Print received queries to the screen

    if data['jsonrpc'] != '2.0':
        response = {'jsonrpc': '2.0', 'error': {'code': -32600, 'message': 'Invalid Request'}, 'id': None}
        print("Sending response:", response)  # Print response to the screen
        return jsonify(response), 400

    # Initialize the result variable
    result = None

    if data['method'] == 'eth_blockNumber':
        result = hex(current_block_number)
        current_block_number += 1  # Increment block number

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
        result = '0x123456'
    
    elif data['method'] == 'eth_chainId':
        result = '0x1A5F0'  # Mainnet

    elif data['method'] == 'eth_getCode':
        address = data['params'][0].lower()  # Convert the address to lower case
        block_number = data['params'][1]  # The block number or tag; you may or may not use this depending on your implementation

        # Example: Fetch the contract code from your database or some storage.
        # In this example, it's hardcoded.
        contract_code = {
            '0xdadf7ac7d0622dedd7e58b4d85d3784cc0c9d0e7': '0x606060405260...'
        }

        result = contract_code.get(address, '0x')  # return '0x' if the address is not a contract

    elif data['method'] == 'eth_getTransactionReceipt':
        tx_hash = data['params'][0]
        print ("Entering eth_getTransactionReceipt"); 
        receipt = {
            'transactionHash': tx_hash,
            'transactionIndex': '0x1',
            'blockHash': '0x' + secrets.token_hex(32),
            'blockNumber': '0x6',
            'from': '0x' + secrets.token_hex(20),
            'to': '0x' + secrets.token_hex(20),
            'cumulativeGasUsed': '0xA',
            'gasUsed': '0xA',
            'contractAddress': None,
            'logs': [],
            'status': '0x1'
        }
        print ("Sending: ", receipt)
        result = receipt

    elif data['method'] == 'eth_getTransactionCount':
        address = data['params'][0].lower()
        block_number = data['params'][1]  # May or may not use this depending on your needs

        # Fetch the nonce from your database or however you're storing nonces
        # For demonstration, we'll just use a hardcoded example
        nonces = {
            "0xc855fd5aa2829799dde83b43ac33651e46f610bb": 10
        }
        result = hex(nonces.get(address, 0))
    
    elif data['method'] == 'eth_getBlockByNumber':
        result = {
            'number': '0x5BAD55',
            'hash': '0x1234567890abcdef',
            'transactions': []
        }
    
    elif data['method'] == 'net_version':
        result = '1'  # Mainnet


    elif data['method'] == 'eth_getBlockByHash':
        block_hash = data['params'][0]
        full_tx = data['params'][1]

        # Fetch block data by block hash.
        # For demonstration purposes, let's use a mock block.
        mock_block = {
            'number': '0x1b4',
            'hash': block_hash,
            'transactions': ['0x123...', '0x124...'] if not full_tx else [
                {
                    'hash': '0x123...',
                    'from': '0xabc...',
                    'to': '0xdef...',
                    'value': '0x1'
                },
                {
                    'hash': '0x124...',
                    'from': '0xghi...',
                    'to': '0xjkl...',
                    'value': '0x2'
                }
            ]
            # ... (other block fields)
        }

        result = mock_block
    
    elif data['method'] == 'eth_gasPrice':
        result = '0x3B9ACA00'  # Example gas price, 1 Gwei in hexadecimal

    elif data['method'] == 'eth_sendRawTransaction':
        print (data)
        raw_tx = data['params'][0]
        print("Raw TX: ", raw_tx)
        try:
            handle_raw_transaction(raw_tx)
            tx_hash = broadcast_transaction(raw_tx)
            result = tx_hash
        except Exception as e:
            result = {'jsonrpc': '2.0', 'error': {'code': -32000, 'message': f'Exception: {e}'}, 'id': data['id']}
            print("Sending response:", result)  # Print response to the screen
            return result

 
    else:
        response = {'jsonrpc': '2.0', 'error': {'code': -32601, 'message': 'Method not found'}, 'id': data.get('id', None)}
        print("Sending response:", response)  # Print response to the screen
        return jsonify(response), 400
    
    response = {
        'jsonrpc': '2.0',
        'id': data.get('id', None),
        'result': result
    }

    print("Sending response:", response)  # Print response to the screen
    return jsonify(response)


def broadcast_transaction(raw_tx):
    # Generate a 64-character long hex string as a fake Ethereum transaction hash
    fake_tx_hash = '0x' + secrets.token_hex(32)
    return fake_tx_hash

from Crypto.Hash import keccak
import rlp
from coincurve import PublicKey

def get_recovered_address(raw_tx):
    w3 = Web3()
    try:
        # Recover the sender's address from the raw transaction
        recovered_address = w3.eth.account.recover_transaction(raw_tx)
        return recovered_address
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def handle_raw_transaction(raw_tx):
    # Decode the raw transaction
    try:
        # Decoding the hex-encoded transaction
        decoded_tx = rlp.decode(bytes.fromhex(raw_tx[2:]), Transaction)

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
