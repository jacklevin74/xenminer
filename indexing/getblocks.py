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
    new_offset = offset + 1400000

    conn = sqlite3.connect('blocks.db', timeout=10)
    c = conn.cursor()

    # Define a function to convert fetched rows into dictionaries
    def convert_rows(rows, id_key):
        return [
            {
                id_key: row[0],
                'hash_to_verify': row[1],
                'key': row[2],
                'account': row[3],
                'date': row[4]
            }
            for row in rows
        ]

    # Fetch and convert records from blocks table
    c.execute("SELECT * FROM blocks ORDER BY block_id ASC LIMIT 100 OFFSET ?", (offset,))
    block_records = convert_rows(c.fetchall(), 'block_id')

    combined_records = block_records

    # Only fetch records from the xuni table if the number is 10000 or greater
    if number >= 10000:
        print ("Fetch xuni ", number, new_offset)
        # Fetch and convert records from xuni table
        c.execute("SELECT * FROM xuni ORDER BY Id ASC LIMIT 100 OFFSET ?", (new_offset,))
        xuni_records = convert_rows(c.fetchall(), 'xuni_id')
        combined_records.extend(xuni_records)  # Append xuni_records to combined_records

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
