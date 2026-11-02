from flask import Flask, request, jsonify, render_template_string
import redis
import json
import threading
import requests
import time

app = Flask(__name__)

# Configure Redis connection
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

@app.route('/store_data', methods=['POST'])
def store_data():
    # just return 200 per Jack
    return jsonify({"message": "Data processed and sent to remote server successfully"}), 200
    data = request.json
    first_block_id = data['first_block_id']
    last_block_id = data['last_block_id']
    final_hash = data['final_hash']
    pubkey = data['pubkey']

    redis_key = f"{first_block_id}_{last_block_id}_{final_hash}"

    # Check if the key already exists in Redis
    if redis_client.exists(redis_key):
        stored_data = json.loads(redis_client.get(redis_key))
        if pubkey not in stored_data['pubkeys']:
            stored_data['counter'] += 1
            stored_data['pubkeys'].append(pubkey)
            redis_client.set(redis_key, json.dumps(stored_data))
    else:
        # Store new data
        new_data = {
            'counter': 1,
            'pubkeys': [pubkey]
        }
        redis_client.set(redis_key, json.dumps(new_data))

    # Send the received data to the remote server
    # L.B. instead of sending it to blockchain-linked component, send it to a lightweight DB-operating one
    threading.Thread(target=send_data, args=(data,)).start()

    return jsonify({"message": "Data processed and sent to remote server successfully"}), 200

@app.route('/show_data', methods=['GET'])
def show_data():
    page = int(request.args.get('page', 1))  # Default to page 1
    per_page = int(request.args.get('per_page', 50))  # Default to 50 entries per page
    cursor = 0  # Start SCAN from the beginning
    all_entries = []

    # Scan through Redis keys in batches
    while True:
        cursor, keys = redis_client.scan(cursor,
                                         count=1000)  # Fetch 1000 keys at a time
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            parts = key.split('_')
            if len(parts) != 3:
                continue  # Skip malformed keys
            first_block_id, last_block_id, final_hash = parts
            entry = json.loads(redis_client.get(key))
            if entry['counter'] > 1:  # Filter records with a counter > 1
                entry.update({
                    'first_block_id': first_block_id,
                    'last_block_id': last_block_id,
                    'final_hash': final_hash
                })
                all_entries.append(entry)

        if len(all_entries) >= (page * per_page):
            break  # Stop scanning once we have enough data for the requested page

        if cursor == 0:
            break  # End of keys

    # Sort by first_block_id in descending order
    all_entries.sort(key=lambda x: int(x['first_block_id']), reverse=True)

    # Paginate results
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_data = all_entries[start_index:end_index]

    # HTML Template
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Voting Ledger</title>
        <style>
            body {
                background-color: #2e2e2e;
                color: #ffffff;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                text-align: center;
                margin-bottom: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #444;
            }
            th {
                background-color: #444;
            }
            tr:nth-child(even) {
                background-color: #3e3e3e;
            }
            tr:hover {
                background-color: #575757;
            }
            .pagination {
                text-align: center;
                margin-top: 20px;
            }
            .pagination a {
                color: white;
                padding: 10px 15px;
                margin: 5px;
                text-decoration: none;
                background-color: #444;
                border-radius: 5px;
            }
            .pagination a:hover {
                background-color: #575757;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>XENBLOCKS Consensus Ledger</h1>
            <table>
                <tr>
                    <th>First Block ID</th>
                    <th>Last Block ID</th>
                    <th>Fingerprint</th>
                    <th>Votes</th>
                    <th>Pubkeys</th>
                </tr>
                {% for entry in data %}
                <tr>
                    <td>{{ entry.first_block_id }}</td>
                    <td>{{ entry.last_block_id }}</td>
                    <td>{{ entry.final_hash }}</td>
                    <td>{{ entry.counter }}</td>
                    <td>{{ entry.pubkeys | join(', ') }}</td>
                </tr>
                {% endfor %}
            </table>

            <div class="pagination">
                {% if page > 1 %}
                    <a href="/show_data?page={{ page - 1 }}&per_page={{ per_page }}">Previous</a>
                {% endif %}
                <a href="/show_data?page={{ page + 1 }}&per_page={{ per_page }}">Next</a>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, data=paginated_data, page=page,
                                  per_page=per_page)

def run_flask_app():
    app.run(host='0.0.0.0', port=5000, debug=False)  # Run without debug=True

def send_data(data):
    # this server is no longer alive. Per Jack, I'm commenting this code out.
    pass

    time.sleep(1)  # Ensure the server has started before sending data
    # LB: instead of 4444 which was storing data on blockchain, we send it directly to votes awards processing stage
    url = 'http://206.72.198.218:4444/'
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("Data processed successfully by remote server")
    else:
        print(f"Failed to process data by remote server: {response.status_code}")
    print(response.text)

if __name__ == '__main__':
    # Run the Flask app in a separate thread
    threading.Thread(target=run_flask_app).start()

    # Example data to be sent
    data_to_send = {
        'first_block_id': '123456',
        'last_block_id': '654321',
        'final_hash': 'abcdef1234567890abcdef1234567890',
        'pubkey': 'pubkey1234567890abcdef'
    }

    # Send data asynchronously using the requests library
    threading.Thread(target=send_data, args=(data_to_send,)).start()


