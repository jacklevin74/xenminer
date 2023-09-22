import sqlite3
from datetime import datetime

def compute_avg_rows_per_minute(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch the last 600 records sorted by created_at
    cursor.execute("SELECT created_at FROM blocks ORDER BY block_id DESC LIMIT 600")
    records = cursor.fetchall()
    
    if len(records) < 2:  # Not enough data to calculate average
        print("Not enough data to calculate average rows per minute.")
        return
    
    # Convert datetime strings to Python datetime objects
    newest_time = datetime.strptime(records[0][0], "%Y-%m-%d %H:%M:%S")
    oldest_time = datetime.strptime(records[-1][0], "%Y-%m-%d %H:%M:%S")
    
    # Calculate time difference in minutes
    time_difference = (newest_time - oldest_time).total_seconds() / 60.0
    
    # Calculate average rows per minute
    avg_rows_per_minute = int(len(records) / time_difference)
    
    print(f"Average rows per minute: {avg_rows_per_minute}")

# Example usage
db_path = "blocks.db"  # Replace with the path to your SQLite database file
compute_avg_rows_per_minute(db_path)

