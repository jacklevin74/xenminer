import sqlite3
import base64
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
                m_value INT,
                t_value INT,
                p_value INT,
                salt_bytes BLOB,
                key BLOB,
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



    def credit_balance(self, block_id, account, hash_to_verify, key, currency_type, amount, created_at):
        try:
            account_bytes = bytes.fromhex(account[2:])
            key_bytes = bytes.fromhex(key)
            m_value, t_value, p_value, salt_bytes = process_argon2id_hash(hash_to_verify)
            self.cursor.execute('''
                INSERT INTO account_balances (block_id, account, m_value, t_value, p_value, salt_bytes,  key, currency_type, amount, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (block_id, account_bytes, m_value, t_value, p_value, salt_bytes,  key_bytes, currency_type, amount, created_at))
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


def process_argon2id_hash(hash):
    # Split the string by '$'
    parts = hash.split('$')

    # Extract the relevant parts
    mtp_part = parts[3]  # 'm=67400,t=1,p=1'
    special_string = parts[4]  # 'WEVOMTAwODIwMjJYRU4'

    # Split the mtp_part to get individual values
    m_value = t_value = p_value = None
    for param in mtp_part.split(','):
        key, value = param.split('=')
        if key == 'm':
            m_value = int(value)
        elif key == 't':
            t_value = int(value)
        elif key == 'p':
            p_value = int(value)

    padding_needed = len(special_string) % 4
    if padding_needed != 0:
        special_string += '=' * (4 - padding_needed)

    # Base64 decode the special string into bytes
    salt_bytes = base64.b64decode(special_string)

    return m_value, t_value, p_value, salt_bytes



def check_and_credit_for_capital_count(block_id, hash_to_verify, key, account_to_update, account_manager, created_at):
    try:
        account_bytes = bytes.fromhex(account_to_update[2:]) 

        last_element = hash_to_verify.split("$")[-1]

        hash_uppercase_only = ''.join(filter(str.isupper, last_element))
        if len(hash_uppercase_only) >= 50:
            account_manager.credit_balance(block_id, account_to_update, hash_to_verify, key, 3, 1, created_at)  # 3 for X.BLK
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
                    record_id = record.get('block_id')
                    created_at = record.get('date')
                    m_value, t_value, p_value, salt_bytes = process_argon2id_hash(hash_to_verify)
                    #print (m_value, t_value, p_value, base64.b64encode(salt_bytes))
                    account_to_update = record.get('account')
                    #print(hash_to_verify, keys, account_to_update)

                    if hash_to_verify:
                        if 'XUNI' in hash_to_verify or bool(re.search('XUNI[0-9]', hash_to_verify)):
                            account_manager.credit_balance(record_id, account_to_update, hash_to_verify,keys, 2, 1, created_at)  # 2 for XUNI
                        elif 'XEN' in hash_to_verify or 'XEN1' in hash_to_verify or 'XEN11' in hash_to_verify:
                            account_manager.credit_balance(record_id, account_to_update, hash_to_verify,keys,1, 1, created_at)  # 1 for XNM
                            check_and_credit_for_capital_count(record_id, hash_to_verify, keys, account_to_update, account_manager, created_at)

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
