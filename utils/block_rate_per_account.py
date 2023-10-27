import sqlite3
import time
import threading

DATABASE_NAME = "blocks.db"  # Update this with your SQLite database name

def populate_account_block_counts():
    # Establish a connection to the SQLite3 database
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()

        # Drop the AccountBlockCounts table if it exists
        c.execute('DROP TABLE IF EXISTS AccountBlockCounts')

        # Create the AccountBlockCounts table
        c.execute('''
        CREATE TABLE AccountBlockCounts (
            account TEXT NOT NULL,
            num_blocks INTEGER NOT NULL
        )
        ''')

        # Use a temporary table to filter the last 24 hours' records
        c.execute('''
        WITH Last24HourBlocks AS (
            SELECT 
                account, 
                created_at
            FROM 
                blocks
            WHERE 
                created_at >= datetime('now', '-1 day')
        )
        INSERT INTO AccountBlockCounts (account, num_blocks)
        SELECT 
            account, 
            COUNT(*) as num_blocks
        FROM 
            Last24HourBlocks
        GROUP BY 
            account
        ORDER BY 
            num_blocks DESC
        ''')

        # Commit the transaction
        conn.commit()

        print("Records successfully inserted into AccountBlockCounts!")

def continuous_population():
    while True:
        print("Populating AccountBlockCounts...")
        populate_account_block_counts()
        time.sleep(300)  # sleep for 5 minutes

if __name__ == "__main__":
    thread = threading.Thread(target=continuous_population)
    thread.start()
    thread.join()  # Optional, if you want the main thread to wait for the spawned thread.

