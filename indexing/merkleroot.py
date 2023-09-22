from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

DATABASE_NAME = "xenminer.db"

def init_db():
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()

@app.route('/send_pow', methods=['POST'])
def send_pow():
    data = request.get_json()

    account_address = data.get('account_address')
    block_id = data.get('block_id')
    merkle_root = data.get('merkle_root')
    key = data.get('key')
    hash_to_verify = data.get('hash_to_verify')

    if not all([account_address, block_id, merkle_root, key, hash_to_verify]):
        return jsonify({"message": "Invalid payload"}), 400

    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            c = conn.cursor()
            c.execute("REPLACE INTO merkleroot2 (block_id, merkleroot_hash, account, key, hash_to_verify) VALUES (?, ?, ?, ?, ?)",
                      (block_id, merkle_root, account_address, key, hash_to_verify))
            conn.commit()
            return jsonify({"message": "POW Record stored successfully!"}), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500 

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

