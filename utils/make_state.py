import sqlite3
import json
import re
import cProfile

class SQLiteAccountManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA synchronous=OFF")
        self.cursor.execute("PRAGMA journal_mode=MEMORY")
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS account_balances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id INTEGER,
                account BLOB,
                currency_type INTEGER,
                amount INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def create_index(self):
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_account_currency
            ON account_balances (account, currency_type)
        ''')
        self.conn.commit()



    def credit_balance(self, block_id, account, currency_type, amount):
        try:
            account_bytes = bytes.fromhex(account[2:])
            
            self.cursor.execute('''
                INSERT INTO account_balances (block_id, account, currency_type, amount)
                VALUES (?, ?, ?, ?)
            ''', (block_id, account_bytes, currency_type, amount))
            self.conn.commit()

            # Check if the row was inserted, if not, it's a collision
            if self.cursor.rowcount == 0:
                print(f"Collision detected for account {account} and currency type {currency_type}")


        except ValueError:
            # If the account is not a valid hexadecimal, ignore it.
            pass


    def get_balance(self, account_bytes, currency_type):
        self.cursor.execute('''
            SELECT amount FROM account_balances
            WHERE account = ? AND currency_type = ?;
        ''', (account_bytes, currency_type))
        result = self.cursor.fetchone()
        return result[0] if result else 0

def check_and_credit_for_capital_count(block_id, hash_to_verify, account_to_update, account_manager):
    try:
        account_bytes = bytes.fromhex(account_to_update[2:]) 

        last_element = hash_to_verify.split("$")[-1]

        hash_uppercase_only = ''.join(filter(str.isupper, last_element))
        if len(hash_uppercase_only) >= 50:
            account_manager.credit_balance(block_id, account_to_update, 3, 1)  # 3 for X.BLK
            return
    except ValueError:
        #print(f"Invalid hexadecimal string: {account_to_update}")
        pass

def generate_superblock_report(db_path, balances_db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    account_manager = SQLiteAccountManager(balances_db_path)

    block_counter = 0
    last_id = 0
    while True:
        cursor.execute("SELECT id, records_json FROM blockchain WHERE id > ? ORDER BY id ASC LIMIT 10000", (last_id,))
        blocks = cursor.fetchall()

        if not blocks:
            break

        for block in blocks:
            block_id = block['id']
            last_id = block_id
            records_json = block['records_json']

            try:
                records = json.loads(records_json)

                for record in records:
                    hash_to_verify = record.get('hash_to_verify')
                    keys = record.get('key')
                    create_at = record.get('created_at)
                    account_to_update = record.get('account')
                    create_at = record.get('created_at)
                    print(hash_to_verify, keys, account_to_update)

                    if hash_to_verify:
                        if 'XUNI' in hash_to_verify or bool(re.search('XUNI[0-9]', hash_to_verify)):
                            account_manager.credit_balance(block_id, account_to_update, 2, 1)  # 2 for XUNI
                        elif 'XEN' in hash_to_verify or 'XEN1' in hash_to_verify or 'XEN11' in hash_to_verify:
                            account_manager.credit_balance(block_id, account_to_update, 1, 1)  # 1 for XNM
                            check_and_credit_for_capital_count(block_id, hash_to_verify, account_to_update, account_manager)

            except json.JSONDecodeError:
                print(f"Error decoding JSON for block ID: {block_id}")

            block_counter += 1

            if block_counter % 100 == 0:
                print(f"Processed {block_counter} blocks")

    account_manager.create_index()

    conn.close()

def main():
    generate_superblock_report('blockchain.db', 'balances6.db')

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.print_stats(sort='time')
