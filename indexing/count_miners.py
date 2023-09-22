import threading
import subprocess
import sqlite3
from datetime import datetime

def count_miners_and_insert_into_db():
    # Run shell commands to count the number of unique miners
    try:
        cmd = "tail -500000 access.log | grep verify | cut -d' ' -f1 | sort | uniq | wc -l"
        total_miners = subprocess.getoutput(cmd)
        total_miners = int(total_miners.strip())  # Convert the output to an integer
        
        # Connect to the difficulty database
        conn = sqlite3.connect('difficulty.db')
        cursor = conn.cursor()

        # Create the miners table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS miners (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_miners INTEGER
        );
        ''')
        
        # Insert the count into the miners table
        cursor.execute("INSERT INTO miners (total_miners) VALUES (?)", (total_miners,))
        conn.commit()
        
        # Close the database connection
        conn.close()
        
        print(f"Inserted {total_miners} into the database at {datetime.now()}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Set up a timer to run the function every 60 seconds
def start_timer():
    threading.Timer(60.0, start_timer).start()
    count_miners_and_insert_into_db()

# Start the timer
start_timer()

