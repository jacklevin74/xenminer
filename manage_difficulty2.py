import sqlite3
import sys
import threading
from datetime import datetime

MIN_DIFFICULTY = 100


def is_within_five_minutes_of_hour():
    timestamp = datetime.now()
    minutes = timestamp.minute
    print("My minutes ", minutes)
    return 0 <= minutes < 5 or 55 <= minutes < 60


def adjust_difficulty():
    # Save stdout to a variable for restoring later
    # original_stdout = sys.stdout

    # Open the log file in append mode
    with open("difficulty.log", "a") as f:
        # Redirect stdout to the file
        sys.stdout = f

        # Connect to the difficulty database
        diff_conn = sqlite3.connect("difficulty.db", timeout=10)
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
        if is_within_five_minutes_of_hour():
            # Check if the latest rate deviates by 20% from 60
            threshold = 70 * 0.2  # 20% of 60
            lower_limit = 70 - threshold
            upper_limit = 70 + threshold
        else:
            # Check if the latest rate deviates by 20% from 60
            threshold = 70 * 0.2  # 20% of 60
            lower_limit = 70 - threshold + 5
            upper_limit = 70 + threshold

        if latest_rate < lower_limit or latest_rate > upper_limit:
            # Update the level based on the latest rate
            new_level = (
                current_level + 1000
                if latest_rate > upper_limit
                else current_level - 2000
            )

            # Enforce minimum difficulty floor
            new_level = max(new_level, MIN_DIFFICULTY)

            # Update the difficulty level in the database
            diff_c.execute("UPDATE difficulty SET level = ? WHERE id=1", (new_level,))
            diff_conn.commit()
            print(f"Difficulty level updated to {new_level}")
            with open("/home/ubuntu/mining/diff2.chain", "w") as file:
                file.write(f"{new_level}\n")

        # Close the connection to the database
        diff_conn.close()

        # Restore original stdout
        # sys.stdout = original_stdout

    # issue randomness every 5 mins

    # diff_conn = sqlite3.connect('difficulty.db', timeout=10)
    # diff_c = diff_conn.cursor()
    # diff_c.execute("UPDATE difficulty SET level = level + ((random() % 51) - 25) WHERE id=1")
    # diff_conn.commit()
    # diff_conn.close()
    # print ("Randomized difficulty")
    # Schedule the next check
    threading.Timer(300, adjust_difficulty).start()


# Initial call to start the loop and reporting
print("Starting the script...")
adjust_difficulty()
