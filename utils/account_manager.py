import sqlite3
from trie import HexaryTrie
import zlib

class SQLiteDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Set PRAGMAs to enhance write performance
        self.cursor.execute("PRAGMA synchronous=OFF")
        self.cursor.execute("PRAGMA journal_mode=MEMORY")
        
        self.setup()

    def setup(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS trie_data (key TEXT UNIQUE, value TEXT)")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS block_meta (block_id INTEGER PRIMARY KEY, root_hash TEXT)")
        self.conn.commit()
    
    def __getitem__(self, key):
        self.cursor.execute("SELECT value FROM trie_data WHERE key=?", (key.hex(),))
        value = self.cursor.fetchone()
        if value:
            # Decompress the value after fetching it from the database
            return zlib.decompress(bytes.fromhex(value[0]))
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        hex_key = key.hex()
        # Compress the value before inserting it into the database
        compressed_value = zlib.compress(value)
        hex_compressed_value = compressed_value.hex()
        self.cursor.execute("INSERT OR IGNORE INTO trie_data (key, value) VALUES (?, ?)", (hex_key, hex_compressed_value))
        self.conn.commit()


    def set_root_hash(self, block_id, root_hash):
        self.cursor.execute("REPLACE INTO block_meta (block_id, root_hash) VALUES (?, ?)", (block_id, root_hash.hex()))
        self.conn.commit()

    def get_root_hash(self, block_id):
        self.cursor.execute("SELECT root_hash FROM block_meta WHERE block_id=?", (block_id,))
        root_hash = self.cursor.fetchone()
        return bytes.fromhex(root_hash[0]) if root_hash else None

    def close(self):
        self.conn.close()

import plyvel
import json
import zlib


class LevelDB:
    def __init__(self, db_path):
        self.db = plyvel.DB(db_path, create_if_missing=True)

    def __getitem2__(self, key):
        # Check if the key is a bytes object; if not, encode it
        encoded_key = key if isinstance(key, bytes) else key.encode()
        value = self.db.get(encoded_key)
        if value is not None:
            # Decompress the value after fetching it from the database
            return zlib.decompress(value)
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        # Check if the key is a bytes object; if not, encode it
        encoded_key = key if isinstance(key, bytes) else key.encode()
        value = self.db.get(encoded_key)
        if value is not None:
            # Return the value directly without decompression
            return value
        else:
            raise KeyError(key)


    def __setitem2__(self, key, value):
        # Check if the key is a bytes object; if not, encode it
        encoded_key = key if isinstance(key, bytes) else key.encode()
        # Compress the value before inserting it into the database
        compressed_value = zlib.compress(value)
        self.db.put(encoded_key, compressed_value)

    def __setitem__(self, key, value):
        encoded_key = key if isinstance(key, bytes) else key.encode()
        self.db.put(encoded_key, value)  # No compression

    def set_root_hash(self, block_id, root_hash):
        # Convert block_id to string and combine with a prefix to differentiate it
        key = f'root_hash_{block_id}'
        # Check if root_hash is a bytes object; if not, encode it
        encoded_root_hash = root_hash if isinstance(root_hash, bytes) else root_hash.encode()
        self.db.put(key.encode(), encoded_root_hash)

    def get_root_hash(self, block_id):
        key = f'root_hash_{block_id}'.encode()
        root_hash = self.db.get(key)
        return root_hash if root_hash else None

    def close(self):
        self.db.close()


class AccountManager:
    def __init__(self, db_path):
        self.db = LevelDB(db_path)
        #self.db = SQLiteDB(db_path)
        self.trie = HexaryTrie(self.db)

    def _construct_key(self, account, currency_type):
        # Construct a key by combining the account and currency_type
        return f"{account}${currency_type}".encode()

    def set_balance(self, account, currency_type, amount, block_id):
        key = self._construct_key(account, currency_type)
        value = str(amount).encode()  # Convert the amount to a string and encode it to bytes
        self.trie[key] = value
        print ("set_balance for account, currency_type, amount, block_id ", account, currency_type, amount, block_id)
        self.db.set_root_hash(block_id, self.trie.root_hash)

    def get_balance(self, account, currency_type):
        key = self._construct_key(account, currency_type)
        amount = self.trie.get(key)
        print ("Account, currency_type, amount ", account, currency_type, amount)
        return int(amount.decode()) if amount else 0

    def credit_balance(self, account, currency_type, amount, block_id):
        current_balance = self.get_balance(account, currency_type)
        new_balance = current_balance + amount
        self.set_balance(account, currency_type, new_balance, block_id)

    def debit_balance(self, account, currency_type, amount, block_id):
        current_balance = self.get_balance(account, currency_type)
        if current_balance < amount:
            raise ValueError(f"Insufficient balance for account {account}, currency {currency_type}")
        new_balance = current_balance - amount
        self.set_balance(account, currency_type, new_balance, block_id)

    def rebuild_trie(self, block_id):
        root_hash = self.db.get_root_hash(block_id)
        self.trie = HexaryTrie(self.db, root_hash)


import unittest
import os

class TestAccountManager(unittest.TestCase):

    def setUp(self):
        # Set up a new AccountManager before each test
        self.manager = AccountManager('test_ptrie.db')  # Use a separate db for testing

    def tearDown(self):
        # Clean up the database file after each test
        self.manager.db.close()
        #os.remove('test_ptrie.db')

    def test_credit_debit_operations(self):
        # Define initial conditions
        accounts = {
            '0xAccount1': {'diamonds': 300, 'gold': 150, 'silver': 100},
            '0xAccount2': {'diamonds': 200, 'gold': 350, 'silver': 450}
        }
        block_id = 420

        # Set initial balances
        for account, assets in accounts.items():
            for asset, balance in assets.items():
                self.manager.set_balance(account, asset, balance, block_id)

        # Perform credit operations
        self.manager.credit_balance('0xAccount1', 'diamonds', 50, block_id)
        self.manager.credit_balance('0xAccount2', 'silver', 50, block_id)

        # Verify the balances after credit
        self.assertEqual(self.manager.get_balance('0xAccount1', 'diamonds'), 350)
        self.assertEqual(self.manager.get_balance('0xAccount2', 'silver'), 500)

        # Perform debit operations
        self.manager.debit_balance('0xAccount1', 'gold', 50, block_id)
        self.manager.debit_balance('0xAccount2', 'silver', 100, block_id)

        # Verify the balances after debit
        self.assertEqual(self.manager.get_balance('0xAccount1', 'gold'), 100)
        self.assertEqual(self.manager.get_balance('0xAccount2', 'silver'), 400)

        # Attempt to debit more than the current balance and check for an exception
        with self.assertRaises(ValueError):
            self.manager.debit_balance('0xAccount1', 'gold', 200, block_id)

    def test_rebuild_trie(self):
        # Additional tests to ensure trie rebuild functionality works as expected
        pass

    # Add more tests as needed for thorough testing


if __name__ == "__main__":
    manager = AccountManager('ptrie_test.db')
    unittest.main()

    # Operations with block_id 1
    initial_balances = {
        '0xSomeAccount1': {"diamonds": 100, "gold": 200, "silver": 300},
        '0xSomeAccount2': {"diamonds": 150, "gold": 250, "silver": 350}
    }

    block_id = 1
    for account, balances in initial_balances.items():
        for currency_type, amount in balances.items():
            manager.set_balance(account, currency_type, amount, block_id)
