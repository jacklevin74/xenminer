import sqlite3
import threading
from datetime import datetime

def compute_avg_rows_per_minute_and_store():
    # Connect to the blocks database
    conn_blocks = sqlite3.connect('blocks.db')
    cursor_blocks = conn_blocks.cursor()

    # Connect to the difficulty database
    conn_diff = sqlite3.connect('difficulty.db')
    cursor_diff = conn_diff.cursor()

    # Create the blockrate table if it doesn't exist
    cursor_diff.execute('''
    CREATE TABLE IF NOT EXISTS blockrate (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATETIME DEFAULT CURRENT_TIMESTAMP,
        rate INTEGER
    );
    ''')
    conn_diff.commit()

    # Fetch the last 600 records from the blocks table
    cursor_blocks.execute("SELECT created_at FROM blocks ORDER BY block_id DESC LIMIT 300")
    records = cursor_blocks.fetchall()

    if len(records) < 2:
        print("Not enough data to calculate average rows per minute.")
        return

    # Convert datetime strings to Python datetime objects
    newest_time = datetime.strptime(records[0][0], "%Y-%m-%d %H:%M:%S")
    oldest_time = datetime.strptime(records[-1][0], "%Y-%m-%d %H:%M:%S")

    # Calculate time difference in minutes
    time_difference = (newest_time - oldest_time).total_seconds() / 60.0

    # Calculate average rows per minute and round it to an integer
    avg_rows_per_minute = int(round(len(records) / time_difference))

    # Insert the calculated rate into the blockrate table in the difficulty database
    cursor_diff.execute("INSERT INTO blockrate (rate) VALUES (?)", (avg_rows_per_minute,))
    conn_diff.commit()

    # Close database connections
    conn_blocks.close()
    conn_diff.close()

    # Log the inserted rate
    print(f"Inserted rate: {avg_rows_per_minute}")

# Set up a timer to run the function every minute
def start_timer():
    threading.Timer(60.0, start_timer).start()
    compute_avg_rows_per_minute_and_store()

# Start the timer
start_timer()

