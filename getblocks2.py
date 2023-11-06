from flask import Flask, jsonify,send_file
from flask_caching import Cache
import sqlite3

app = Flask(__name__)


@app.route('/total_blocks', methods=['GET'])
def total_blocks():
    conn = sqlite3.connect('blocks.db')
    c = conn.cursor()

    # Query to get the total count of blocks from the `allblocks` table
    c.execute('SELECT COUNT(*) FROM allblocks2')

    result = c.fetchone()
    total_blocks = result[0] if result else 0

    conn.close()

    return jsonify({'total_blocks': total_blocks})


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


@app.route('/getallblocks/<int:page>', methods=['GET'])
def get_records_all(page):
    offset = (page - 1) * 100  # Calculate the offset

    conn = sqlite3.connect('blocks.db', timeout=10)
    c = conn.cursor()

    # Fetch 100 records based on the page number from the allblocks table
    c.execute("SELECT * FROM allblocks LIMIT 100 OFFSET ?", (offset,))
    rows = c.fetchall()

    conn.close()

    # Convert fetched rows into a list of dictionaries
    records = []
    for row in rows:
        record = {
            'id': row[0],  # Assuming the autoincrement id is at index 0
            'hash_to_verify': row[3],
            'key': row[4],
            'account': row[5],
            'date': row[6] or row[7]  # Adjust the index if necessary
        }
        # Check if block_id is present
        if row[1] is not None:
            record['block_id'] = row[1]
        # Check if xuni_id is present
        if row[2] is not None:
            record['xuni_id'] = row[2]
        
        records.append(record)

    return jsonify(records)

@app.route('/download', methods=['GET'])
def download_file():
    path = f"/root/dev/blockchain.db.arc.gz"
    return send_file(path, as_attachment=True)



@app.route('/getallblocks2/<int:page>', methods=['GET'])
def get_records_all2(page):

    records = get_records(page)

    return jsonify(records)

def get_records(page):
    id_min = (page - 1) * 100  # Calculate the minimum id for this page
    id_max = page * 100     # Calculate the maximum id for this page

    conn = sqlite3.connect('blocks.db', timeout=10)
    c = conn.cursor()


    # Query to fetch records based on block_id range
    c.execute("""
        SELECT allblocks2.id, blocks.block_id as block_id, blocks.hash_to_verify, blocks.key, blocks.account, blocks.created_at, 'blocks' as label
        FROM allblocks2
        JOIN blocks ON allblocks2.block_id = blocks.block_id
        WHERE allblocks2.id > ? AND allblocks2.id <= ?
    """, (id_min, id_max,))
    rows_block_id = c.fetchall()

    # Query to fetch records based on xuni_id range
    c.execute("""
        SELECT allblocks2.id, xuni.Id as xuni_id, xuni.hash_to_verify, xuni.key, xuni.account, xuni.created_at, 'xuni' as label
        FROM allblocks2
        JOIN xuni ON allblocks2.xuni_id = xuni.Id
        WHERE allblocks2.id > ? AND allblocks2.id <= ?
    """, (id_min, id_max,))
    rows_xuni_id = c.fetchall()

    conn.close()

    # Print debug information
    print(f'Fetched {len(rows_block_id)} records with block_id and {len(rows_xuni_id)} records with xuni_id')

    # Convert fetched rows into a list of dictionaries
    records = []
    block_id_count = 0  # Initialize block_id_count
    xuni_id_count = 0   # Initialize xuni_id_count

    for row in rows_block_id + rows_xuni_id:
        #print(row)
        record = {
            'id': row[0],
            'hash_to_verify': row[2],
            'key': row[3],
            'account': row[4],
            'date': row[5]  # Adjust the index if necessary
        }

        if row[6] == 'blocks':  # Assuming label is the 7th element in the row
            record['block_id'] = row[1]
            block_id_count += 1
        elif row[6] == 'xuni':  # Assuming label is the 7th element in the row
            record['xuni_id'] = row[1]
            xuni_id_count += 1

        records.append(record)

    print(f'block_id count: {block_id_count}')
    print(f'xuni_id count: {xuni_id_count}')
    return records
