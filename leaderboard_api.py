from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from gpage import get_difficulty

app = Flask(__name__)
CORS(app, resources={r"/leaderboard*": {"origins": "*"}})

def fetch_cache_data(limit, offset):
    with sqlite3.connect('cache.db', timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cache_table ORDER BY total_blocks DESC LIMIT ? OFFSET ?", (limit, offset))
        return cursor.fetchall()

def fetch_latest_rate():
    with sqlite3.connect('difficulty.db', timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT rate FROM blockrate ORDER BY id DESC LIMIT 1")
        return cursor.fetchone()

def fetch_latest_miners():
    with sqlite3.connect('difficulty.db', timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT total_miners FROM miners ORDER BY id DESC LIMIT 1")
        return cursor.fetchone()

def fetch_total_blocks():
    with sqlite3.connect('blocks.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT block_id FROM blocks ORDER BY block_id DESC LIMIT 1')
        return cursor.fetchone()

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    limit = int(request.args.get('limit', 500))
    offset = int(request.args.get('offset', 0))

    difficulty = get_difficulty()
    cache_data = fetch_cache_data(limit, offset)
    latest_rate = fetch_latest_rate()
    latest_miners = fetch_latest_miners()
    total_blocks = fetch_total_blocks()

    latest_rate = latest_rate[0] if latest_rate else 0
    latest_miners = latest_miners[0] if latest_miners else 0
    total_blocks = total_blocks[0] if total_blocks else None

    miners = [
        {
            'rank': i + 1 + offset,
            'account': r[0].strip(),
            'blocks': r[1],
            'hashRate': round(r[2], 2),
            'superBlocks': r[3]
        }
        for i, r in enumerate(cache_data)
    ]

    return jsonify({
        "totalHashRate": latest_rate,
        "totalMiners": latest_miners,
        "totalBlocks": total_blocks,
        "difficulty": difficulty,
        "miners": miners
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5566, debug=True)
