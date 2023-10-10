import sqlite3
from trie import HexaryTrie

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
            value = str(balance).encode()
            self.trie[key] = value
        self.db.set_root_hash(block_id, self.trie.root_hash)

    def get_balance(self, account):
        key = account.encode()
        return int(self.trie.get(key) or b'0')

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



if __name__ == "__main__":
    manager = AccountManager('ptrie.db')

    # Operations with block_id 1
    manager.set_balances({
        '0xSomeAccount1': 100,
        '0xSomeAccount2': 200
    }, 1)

    # Create 100 transactions
    for i in range(2, 102):  # start from block_id 2 and go up to 101
        manager.debit_balances({
            '0xSomeAccount1': 1,
            '0xSomeAccount2': 1
        }, i)

    # Loop to rebuild the trie for block_ids from 50 to 101 and print the balance for each
    for i in range(50, 102):
        # Rebuild trie with the current block_id
        manager.rebuild_trie(i)

        # Print the expected balance for both accounts
        print(f"Balance for block_id {i} (Account1):", manager.get_balance('0xSomeAccount1'))
        print(f"Balance for block_id {i} (Account2):", manager.get_balance('0xSomeAccount2'))

    # Close the db connection
    manager.db.close()

