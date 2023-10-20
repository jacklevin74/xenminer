import sqlite3
import json
import re
import cProfile
import os

CHECKPOINT_FILE = 'balances_checkpoint'


def get_last_block_id(cursor):
    cursor.execute('SELECT id FROM blockchain ORDER BY id DESC LIMIT 1;')
    result = cursor.fetchone()
    return result[0] if result else None


def save_checkpoint(block_id):
    with open(CHECKPOINT_FILE, 'w') as file:
        file.write(str(block_id))


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as file:
            return int(file.read().strip())
    return 0  # No checkpoint file found


class SQLiteAccountManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.balances = {}  # In-memory storage of balances
        self.cursor.execute("PRAGMA synchronous=OFF")
        self.cursor.execute("PRAGMA journal_mode=MEMORY")
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS account_balances (
                account TEXT,
                currency_type INTEGER,
                amount INTEGER,
                PRIMARY KEY (account, currency_type)
            )
        ''')
        self.conn.commit()

    def credit_balance(self, account, currency_type, amount):
        key = (account, currency_type)
        if key not in self.balances:
            self.balances[key] = self.get_balance(account, currency_type)
        self.balances[key] += amount

    def get_balance(self, account, currency_type):
        self.cursor.execute('''
            SELECT amount FROM account_balances
            WHERE account = ? AND currency_type = ?;
        ''', (account, currency_type))
        result = self.cursor.fetchone()
        return result[0] if result else 0

    def save_balances_to_db(self):
        for (account, currency_type), amount in self.balances.items():
            self.cursor.execute('''
                INSERT INTO account_balances (account, currency_type, amount)
                VALUES (?, ?, ?)
                ON CONFLICT(account, currency_type) DO UPDATE SET
                amount = excluded.amount;
            ''', (account, currency_type, amount))
        self.conn.commit()


def check_and_credit_for_capital_count(hash_to_verify, account_to_update, account_manager):
    # Remove all lowercase characters from the string
    hash_uppercase_only = ''.join(filter(str.isupper, hash_to_verify))

    # If the length of the remaining string is 65 or more, it means we have at least 65 uppercase letters.
    if len(hash_uppercase_only) >= 65:
        account_manager.credit_balance(account_to_update, 3, 1)  # 3 for X.BLK


def generate_superblock_report(db_path, balances_db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()


    account_manager = SQLiteAccountManager(balances_db_path)

    block_counter = 0
    last_id = load_checkpoint()

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
                    account_to_update = record.get('account')

                    if hash_to_verify and 'WEVOMTAwODIwMjJYRU4' in hash_to_verify:
                        if 'XUNI' in hash_to_verify or bool(re.search('XUNI[0-9]', hash_to_verify)):
                            account_manager.credit_balance(account_to_update, 2, 1)  # 2 for XUNI
                        elif 'XEN' in hash_to_verify or 'XEN1' in hash_to_verify or 'XEN11' in hash_to_verify:
                            account_manager.credit_balance(account_to_update, 1, 10)  # 1 for XNM. Reward 10x for each XEN{1,11} block
                            check_and_credit_for_capital_count(hash_to_verify, account_to_update, account_manager)

            except json.JSONDecodeError:
                print(f"Error decoding JSON for block ID: {block_id}")

            block_counter += 1

            if block_counter % 1000 == 0:  # Save to DB at intervals
                account_manager.save_balances_to_db()
                print(f"Processed {block_counter} blocks")

    # Save any remaining balances at the end
    account_manager.save_balances_to_db()
    last_block_id = get_last_block_id(cursor)
    save_checkpoint (last_block_id)
    conn.close()

def main():
    # Replace 'your_database_path.db' and 'your_balances_database_path.db' with your actual database paths
    generate_superblock_report('blockchain.db', 'balances.db')

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()  # start profiling
    main()
    profiler.disable()  # end profiling
    profiler.print_stats(sort='time')

