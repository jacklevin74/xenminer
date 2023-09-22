import sqlite3
import threading

def adjust_difficulty():
    # Connect to the difficulty database
    diff_conn = sqlite3.connect('difficulty.db', timeout=10)
    diff_c = diff_conn.cursor()
    
    # Get the latest rate from blockrate table
    diff_c.execute("SELECT rate FROM blockrate ORDER BY id DESC LIMIT 1")
    latest_rate = diff_c.fetchone()
    if latest_rate:
        latest_rate = latest_rate[0]
        print(f"Latest rate is {latest_rate}")
    else:
        print("No rate found")
        return

    # Get the current level of difficulty
    diff_c.execute("SELECT level FROM difficulty WHERE id=1")
    current_level = diff_c.fetchone()
    if current_level:
        current_level = current_level[0]
        print(f"Current difficulty level is {current_level}")
    else:
        print("No difficulty level found")
        return
    
    # Check if the latest rate deviates by 20% from 60
    threshold = 60 * 0.2  # 20% of 60
    lower_limit = 60 - threshold
    upper_limit = 60 + threshold
    
    if latest_rate < lower_limit or latest_rate > upper_limit:
        # Update the level based on the latest rate
        new_level = current_level + 100 if latest_rate > upper_limit else current_level - 100
        
        # Update the difficulty level in the database
        diff_c.execute("UPDATE difficulty SET level = ? WHERE id=1", (new_level,))
        diff_conn.commit()
        print(f"Difficulty level updated to {new_level}")
            
    # Close the connection to the database
    diff_conn.close()

    # Schedule the next check
    threading.Timer(300, adjust_difficulty).start()

# Initial call to start the loop and reporting
print("Starting the script...")
adjust_difficulty()

