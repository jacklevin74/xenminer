import sqlite3
import sys
import time
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

BLOCKS_DB = "file:blocks.db"
CACHE_DB = "cache.db"
SIGNERS_DB = "file:signer_data.db?immutable=1"

EPOCH_SQL = """
    CASE
        WHEN block_id >= 63158400 THEN 3
        WHEN block_id > 29818420 THEN 2
        ELSE 1
    END
"""

# The leaderboard query. It reads the counts that add_new_rows() keeps in cache.db.
LEADERBOARD_SQL = """
WITH grouped_blocks_by_epoch AS (
    SELECT account, epoch, blocks AS blocks_per_epoch
    FROM cache.blocks_by_epoch
),

xuni_counts AS (
    SELECT account, xuni AS total_xuni
    FROM cache.xuni_by_account
),

account_performance as (SELECT
    b.account,
    SUM(blocks_per_epoch) AS total_blocks,
    SUM(blocks_per_epoch * POWER(10, 19) / POWER(2, epoch - 1)) AS xnm,
    COALESCE(sb.super_block_count, 0) AS super_blocks,
    COALESCE(x.total_xuni, 0) AS total_xuni
FROM grouped_blocks_by_epoch b
    LEFT JOIN super_blocks sb ON b.account = sb.account
    LEFT JOIN xuni_counts x ON b.account = x.account
GROUP BY b.account)

SELECT
    ap.account,
    ROW_NUMBER() OVER (ORDER BY total_blocks DESC, super_blocks DESC, total_xuni DESC, ap.account DESC) AS rank,
    total_blocks,
    super_blocks,
    xnm,
    super_blocks * power(10, 18) AS xblk,
    total_xuni * power(10, 18) AS xuni,
    100000 AS hashes_per_second,
    sn.solanaPubkey
FROM account_performance ap
    LEFT OUTER JOIN signers.signers_normalized sn ON LOWER(sn.ethAddress) = ap.account
ORDER BY rank
"""


def create_tables(cache_conn):
    cache_cursor = cache_conn.cursor()
    cache_cursor.execute("""
    CREATE TABLE IF NOT EXISTS cache_table (
        account TEXT PRIMARY KEY,
        total_blocks INTEGER,
        hashes_per_second REAL,
        super_blocks INTEGER,
        rank INTEGER DEFAULT 0,
        xnm BIGINT DEFAULT 0,
        xblk BIGINT DEFAULT 0,
        xuni BIGINT DEFAULT 0,
        sol_address TEXT
    )""")
    for column in ("rank INTEGER DEFAULT 0", "xnm BIGINT DEFAULT 0", "xblk BIGINT DEFAULT 0",
                   "xuni BIGINT DEFAULT 0", "sol_address TEXT"):
        try:
            cache_cursor.execute(f"ALTER TABLE cache_table ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass

    # Running counts, and the highest block_id and xuni Id that they include.
    cache_cursor.execute("""
    CREATE TABLE IF NOT EXISTS blocks_by_epoch (
        account TEXT,
        epoch INTEGER,
        blocks INTEGER,
        PRIMARY KEY (account, epoch)
    )""")
    cache_cursor.execute("""
    CREATE TABLE IF NOT EXISTS xuni_by_account (
        account TEXT PRIMARY KEY,
        xuni INTEGER
    )""")
    cache_cursor.execute("""
    CREATE TABLE IF NOT EXISTS cache_progress (
        name TEXT PRIMARY KEY,
        last_id INTEGER
    )""")
    cache_conn.commit()


def reset_counts(cache_conn):
    """Delete the running counts. The next pass counts all rows again."""
    cache_conn.execute("DELETE FROM blocks_by_epoch")
    cache_conn.execute("DELETE FROM xuni_by_account")
    cache_conn.execute("DELETE FROM cache_progress")
    cache_conn.commit()


def last_id(cache_conn, name):
    row = cache_conn.execute("SELECT last_id FROM cache_progress WHERE name = ?", (name,)).fetchone()
    return row[0] if row else 0


def add_new_rows(original_conn, cache_conn, upto_block=None, upto_xuni=None):
    """Add the blocks and xuni above the stored IDs to the running counts.

    New rows always get a higher ID, so the rows above the stored IDs are the new rows.
    The counts and the new IDs are written in one transaction.
    """
    lo_block = last_id(cache_conn, "blocks")
    lo_xuni = last_id(cache_conn, "xuni")

    # One read transaction, so the maximums and the counts come from the same snapshot.
    original_conn.execute("BEGIN")
    try:
        hi_block = upto_block if upto_block is not None else original_conn.execute(
            "SELECT IFNULL(MAX(block_id), 0) FROM blocks").fetchone()[0]
        hi_xuni = upto_xuni if upto_xuni is not None else original_conn.execute(
            "SELECT IFNULL(MAX(Id), 0) FROM xuni").fetchone()[0]
        blocks = original_conn.execute(f"""
            SELECT LOWER(account), {EPOCH_SQL}, COUNT(*)
            FROM blocks
            WHERE block_id > ? AND block_id <= ?
            GROUP BY 1, 2
        """, (lo_block, hi_block)).fetchall()
        xuni = original_conn.execute("""
            SELECT LOWER(account), COUNT(*)
            FROM xuni
            WHERE Id > ? AND Id <= ?
            GROUP BY 1
        """, (lo_xuni, hi_xuni)).fetchall()
    finally:
        original_conn.commit()

    cache_conn.executemany("""
        INSERT INTO blocks_by_epoch (account, epoch, blocks) VALUES (?, ?, ?)
        ON CONFLICT (account, epoch) DO UPDATE SET blocks = blocks + excluded.blocks
    """, blocks)
    cache_conn.executemany("""
        INSERT INTO xuni_by_account (account, xuni) VALUES (?, ?)
        ON CONFLICT (account) DO UPDATE SET xuni = xuni + excluded.xuni
    """, xuni)
    cache_conn.executemany("INSERT OR REPLACE INTO cache_progress (name, last_id) VALUES (?, ?)",
                           [("blocks", hi_block), ("xuni", hi_xuni)])
    cache_conn.commit()

    new_blocks = sum(row[2] for row in blocks)
    new_xuni = sum(row[1] for row in xuni)
    logging.info(f"Added {new_blocks} blocks (block_id {lo_block}-{hi_block}) "
                 f"and {new_xuni} xuni (Id {lo_xuni}-{hi_xuni}).")


def rebuild_cache_table(original_conn, cache_conn):
    original_conn.execute(f"ATTACH DATABASE '{CACHE_DB}' AS cache")
    try:
        rows = original_conn.execute(LEADERBOARD_SQL).fetchall()
    finally:
        original_conn.execute("DETACH DATABASE cache")

    cache_conn.executemany("""
    INSERT OR REPLACE INTO cache_table
        (account, rank, total_blocks, super_blocks, xnm, xblk, xuni, hashes_per_second, sol_address)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    cache_conn.commit()
    return len(rows)


def recreate_cache_table():
    logging.info("Generating cache table")

    try:
        original_conn = sqlite3.connect(BLOCKS_DB, uri=True)
    except sqlite3.OperationalError:
        logging.error("Failed to connect to blocks.db. Please ensure the database is available.")
        return

    try:
        cache_conn = sqlite3.connect(CACHE_DB)
    except sqlite3.OperationalError:
        logging.error("Failed to connect to cache.db. Please ensure the database is available.")
        original_conn.close()
        return

    try:
        create_tables(cache_conn)
        original_conn.execute(f"ATTACH DATABASE '{SIGNERS_DB}' AS signers")
        add_new_rows(original_conn, cache_conn)
        num_rows = rebuild_cache_table(original_conn, cache_conn)
        logging.info(f"Cache table updated. {num_rows} rows were fetched and inserted.")
    except sqlite3.OperationalError as e:
        cache_conn.rollback()
        logging.error(f"SQLite error: {e}. The next pass tries again.")
    finally:
        original_conn.close()
        cache_conn.close()


def run_db_operations():
    while True:
        recreate_cache_table()
        time.sleep(300)


if __name__ == "__main__":
    logging.info("Starting make_cache")
    if "--rebuild" in sys.argv:
        # Count all rows again, for example after rows were inserted with explicit IDs.
        conn = sqlite3.connect(CACHE_DB)
        create_tables(conn)
        reset_counts(conn)
        conn.close()
        logging.info("Deleted the running counts. The first pass counts all rows.")
    t = threading.Thread(target=run_db_operations, daemon=True)
    t.start()
    t.join()
