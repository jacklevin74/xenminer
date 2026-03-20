import logging
import sqlite3
import threading
from datetime import datetime

MIN_DIFFICULTY = 100

logging.basicConfig(
    filename="difficulty.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def is_within_five_minutes_of_hour():
    timestamp = datetime.now()
    minutes = timestamp.minute
    logger.debug("Current minutes: %d", minutes)
    return 0 <= minutes < 5 or 55 <= minutes < 60


def adjust_difficulty():
    try:
        diff_conn = sqlite3.connect("difficulty.db", timeout=10)
        diff_c = diff_conn.cursor()

        diff_c.execute("SELECT rate FROM blockrate ORDER BY id DESC LIMIT 1")
        latest_rate = diff_c.fetchone()
        if latest_rate:
            latest_rate = latest_rate[0]
            logger.info("Latest rate is %s", latest_rate)
        else:
            logger.warning("No rate found")
            diff_conn.close()
            return

        diff_c.execute("SELECT level FROM difficulty WHERE id=1")
        current_level = diff_c.fetchone()
        if current_level:
            current_level = current_level[0]
            logger.info("Current difficulty level is %s", current_level)
        else:
            logger.warning("No difficulty level found")
            diff_conn.close()
            return

        if is_within_five_minutes_of_hour():
            threshold = 70 * 0.2
            lower_limit = 70 - threshold
            upper_limit = 70 + threshold
        else:
            threshold = 70 * 0.2
            lower_limit = 70 - threshold + 5
            upper_limit = 70 + threshold

        if latest_rate < lower_limit or latest_rate > upper_limit:
            new_level = (
                current_level + 1000
                if latest_rate > upper_limit
                else current_level - 2000
            )

            new_level = max(new_level, MIN_DIFFICULTY)

            diff_c.execute("UPDATE difficulty SET level = ? WHERE id=1", (new_level,))
            diff_conn.commit()
            logger.info("Difficulty level updated to %s", new_level)
            with open("/home/ubuntu/mining/diff2.chain", "w") as file:
                file.write(f"{new_level}\n")

        diff_conn.close()
    except Exception:
        logger.exception("Error adjusting difficulty")
    finally:
        threading.Timer(300, adjust_difficulty).start()


logger.info("Starting the script...")
adjust_difficulty()
