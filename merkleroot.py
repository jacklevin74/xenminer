from flask import Flask, request, jsonify
import json
import requests
import time
from passlib.hash import argon2
import hashlib
from random import choice, randrange
import string
import threading
import re
import sqlite3
import logging

app = Flask(__name__)

DATABASE_NAME = "xenminer.db"

# Set up logging to file
logging.basicConfig(filename='pow.err',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')


def init_db():
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()

@app.route('/send_pow', methods=['POST'])
def send_pow():
    data = request.get_json()
    app.logger.info("Invalid Payload, missng ", data)
    keys_list = list(data.keys())
    print (keys_list)
    print (data)
    account_address = data.get('account_address')
    block_id = data.get('block_id')
    merkle_root = data.get('merkle_root')
    key = data.get('key')
    mykey = data.get('key')
    hash_to_verify = data.get('hash_to_verify')
    print (account_address, block_id, merkle_root, key,mykey, hash_to_verify)

    if not all([account_address, block_id, merkle_root, key, hash_to_verify]):
        app.logger.info("Invalid Payload, missng ", account_address, block_id, merkle_root, key, hash_to_verify)
        return jsonify({"message": "Invalid payload - However this error isn't critical"}), 400

    # Verifying the hash_to_verify and key using Argon2
    try:
        if argon2.verify(key, hash_to_verify):
            with sqlite3.connect(DATABASE_NAME) as conn:
                c = conn.cursor()
                c.execute(
                    "REPLACE INTO merkleroot2 (block_id, merkleroot_hash, account, key, hash_to_verify) VALUES (?, ?, ?, ?, ?)",
                    (block_id, merkle_root, account_address, key, hash_to_verify)
                )
                conn.commit()
                app.logger.info("POW Record stored successfully")
                return jsonify({"message": "POW Record stored successfully!"}), 200
        else:
            app.logger.info("Invalid key provided")
            return jsonify({"message": "Invalid key provided"}), 400

    except Exception as e:
        app.logger.info("Exception: ", str(e))
        return jsonify({"message": str(e)}), 500
