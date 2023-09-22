from flask import Flask, jsonify
from flask_caching import Cache
import sqlite3

app = Flask(__name__)

@app.route('/getblocks/lastblock', methods=['GET'])
def get_last_block():
    conn = sqlite3.connect('blocks.db', timeout=10)
    c = conn.cursor()

    # Get the last ID in the table
    c.execute("SELECT MAX(block_id) FROM blocks")
    last_id = c.fetchone()[0]

    last_full_page_id = (last_id // 100) * 100  # Last full page of 100 records

    if last_id % 100 == 0:
        offset = last_full_page_id  # If exactly divisible, get the last full page
    else:
        offset = last_full_page_id - 100  # Otherwise go to the previous full page

    # Fetch the last or previous 100 records based on the offset
    c.execute("SELECT * FROM blocks WHERE block_id > ? ORDER BY block_id ASC LIMIT 100", (offset,))
    rows = c.fetchall()

    conn.close()

    # Convert fetched rows into a list of dictionaries
    records = []
    for row in rows:
        record = {
            'block_id': row[0],
            'hash_to_verify': row[1],
            'key': row[2],
            'account': row[3],
            'date': row[4]
        }
        records.append(record)

    return jsonify(records)

@app.route('/getblocks/all/<int:number>', methods=['GET'])
def get_combined_records(number):
    offset = (number - 1) * 100  # Calculate the offset based on the number input

    conn = sqlite3.connect('blocks.db', timeout=10)
    c = conn.cursor()

    # Fetch 100 records from the blocks table based on the offset
    c.execute("SELECT * FROM blocks ORDER BY block_id ASC LIMIT 100 OFFSET ?", (offset,))
    block_rows = c.fetchall()

    combined_records = []

    # Convert fetched rows from blocks table into dictionaries and append to combined_records
    for row in block_rows:
        record = {
            'block_id': row[0],
            'hash_to_verify': row[1],
            'key': row[2],
            'account': row[3],
            'date': row[4]
        }
        combined_records.append(record)

    # Only fetch records from the xuni table if the number is 10000 or greater
    if number >= 1:
        # Fetch 100 records from the xuni table based on the offset
        c.execute("SELECT * FROM xuni ORDER BY Id ASC LIMIT 100 OFFSET ?", (offset,))
        xuni_rows = c.fetchall()

        # Convert fetched rows from xuni table into dictionaries and append to combined_records
        for row in xuni_rows:
            record = {
                'xuni_id': row[0],
                'hash_to_verify': row[1],
                'key': row[2],
                'account': row[3],
                'date': row[4]
            }
            combined_records.append(record)

    conn.close()
    return jsonify(combined_records)


@app.route('/getblocks/<int:page>', methods=['GET'])
def get_records(page):
    offset = (page - 1) * 100  # Calculate the offset

    conn = sqlite3.connect('blocks.db', timeout=10)
    c = conn.cursor()

    # Fetch 100 records based on the page number
    c.execute("SELECT * FROM blocks LIMIT 100 OFFSET ?", (offset,))
    rows = c.fetchall()

    conn.close()

    # Convert fetched rows into a list of dictionaries
    records = []
    for row in rows:
        record = {
            'block_id': row[0],
            'hash_to_verify': row[1],
            'key': row[2],
            'account': row[3],
            'date': row[4]
        }
        records.append(record)

    return jsonify(records)

