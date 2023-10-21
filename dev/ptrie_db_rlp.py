import sqlite3
from trie import HexaryTrie
import rlp

class SQLiteDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup()

    def setup(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS trie_data (key TEXT UNIQUE, value TEXT)")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS block_meta (block_id INTEGER PRIMARY KEY, root_hash TEXT)")
        self.conn.commit()

    def __getitem__(self, key):
        self.cursor.execute("SELECT value FROM trie_data WHERE key=?", (key.hex(),))
        value = self.cursor.fetchone()
        if value:
            return bytes.fromhex(value[0])
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        hex_key = key.hex()
        hex_value = value.hex()
        self.cursor.execute("INSERT OR IGNORE INTO trie_data (key, value) VALUES (?, ?)", (hex_key, hex_value))
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

class AccountManager:
    def __init__(self, db_path):
        self.db = SQLiteDB(db_path)
        self.trie = HexaryTrie(self.db)

    def set_balances(self, balances, block_id):
        for account, balance in balances.items():
            key = account.encode()
            # Here we're encoding the balance as an RLP item before storing it
            encoded_balance = rlp.encode(balance)
            self.trie[key] = encoded_balance
        self.db.set_root_hash(block_id, self.trie.root_hash)

    def get_balance(self, account):
        key = account.encode()
        encoded_balance = self.trie.get(key)
        if encoded_balance:
            # If we find an encoded balance, decode it
            balance = rlp.decode(encoded_balance, sedes=rlp.sedes.big_endian_int)
            return balance
        else:
            return 0  # Return a default balance of 0 if no balance found

    def credit_balances(self, credits, block_id):
        new_balances = {}
        for account, amount in credits.items():
            balance = self.get_balance(account)
            new_balances[account] = balance + amount
        self.set_balances(new_balances, block_id)

    def debit_balances(self, debits, block_id):
        new_balances = {}
        for account, amount in debits.items():
            balance = self.get_balance(account)
            if balance < amount:
                raise ValueError(f"Insufficient balance for account {account}")
            new_balances[account] = balance - amount
        self.set_balances(new_balances, block_id)

    def rebuild_trie(self, block_id):
        root_hash = self.db.get_root_hash(block_id)
        self.trie = HexaryTrie(self.db, root_hash)

import time
if __name__ == "__main__":
    manager = AccountManager('ptrie.db')

    # Operations with block_id 1
    manager.set_balances({
        '0xSomeAccount1': 100,
        '0xSomeAccount2': 200
    }, 1)

    # Start measuring time for credit_balances loop
    start_time_credit_balances = time.time()

    # Create 100 transactions
    for i in range(2, 300000):
        manager.credit_balances({
            '0xSomeAccount1': 1,
            '0xSomeAccount2': 1
        }, i)

        # Print progress every 10,000 iterations
        if i % 10000 == 0:
            print(f"credit_balances progress: {i}/1000000")

    # Calculate and print the rate for credit_balances loop
    end_time_credit_balances = time.time()
    duration_credit_balances = end_time_credit_balances - start_time_credit_balances
    rate_credit_balances = 1000000 / duration_credit_balances
    print(f"credit_balances runs per second: {rate_credit_balances:.2f}")

    # Start measuring time for rebuild_trie loop
    start_time_rebuild_trie = time.time()

    # Loop to rebuild the trie for block_ids from 50 to 101 and print the balance for each
    for i in range(50, 300000):
        manager.rebuild_trie(i)

        # Print progress every 10,000 iterations
        if (i - 50) % 10000 == 0:
            print(f"rebuild_trie progress: {i}/1000000")
        
        #print(f"Balance for block_id {i} (Account1):", manager.get_balance('0xSomeAccount1'))
        #print(f"Balance for block_id {i} (Account2):", manager.get_balance('0xSomeAccount2'))

    # Calculate and print the rate for rebuild_trie loop
    end_time_rebuild_trie = time.time()
    duration_rebuild_trie = end_time_rebuild_trie - start_time_rebuild_trie
    rate_rebuild_trie = (1000000 - 50) / duration_rebuild_trie
    print(f"rebuild_trie runs per second: {rate_rebuild_trie:.2f}")

    # Close the db connection
    manager.db.close()
