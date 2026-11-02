from flask import Flask, jsonify, render_template
import json
import redis
from datetime import datetime

app = Flask(__name__)

# Configure Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def truncate_last_octet(ip):
    return '.'.join(ip.split('.')[:3] + ['xxx'])

def parse_timestamp(node_data):
    if 'timestamp' in node_data:
        try:
            return datetime.strptime(node_data['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            return datetime.min
    return datetime.min


@app.route('/nodes')
def nodes():
    all_keys = redis_client.keys('*')
    nodes_data = []

    for key in all_keys:
        # Skip keys with excluded prefixes
        if key.startswith('x1:hashes') or key.startswith('x1:keys'):
            continue

        try:
            data = redis_client.get(key)
            if data:
                node_data = json.loads(data)  # Ensure data is JSON
                if isinstance(node_data, dict):  # Check if node_data is a dictionary
                    node_data['node'] = truncate_last_octet(key)
                    nodes_data.append(node_data)
        except (redis.exceptions.ResponseError, json.JSONDecodeError):
            continue  # Skip keys that cause errors

    # Sort nodes by timestamp
    nodes_data.sort(key=parse_timestamp, reverse=True)

    total_nodes = len(nodes_data)
    return render_template('nodes.html', nodes=nodes_data, total_nodes=total_nodes)


def get_node_data():
    all_keys = redis_client.keys('*')
    nodes_data = []

    for key in all_keys:
        if key.startswith('x1:hashes') or key.startswith('x1:keys'):
            continue  # Skip keys with excluded prefixes

        try:
            data = redis_client.get(key)
            if data:
                node_data = json.loads(data)  # Ensure data is JSON
                if isinstance(node_data, dict):  # Check if node_data is a dictionary
                    node_data['node'] = truncate_last_octet(key)
                    nodes_data.append(node_data)
        except (redis.exceptions.ResponseError, json.JSONDecodeError):
            continue  # Skip keys that cause errors

    nodes_data.sort(key=parse_timestamp, reverse=True)
    return nodes_data

@app.route('/nodes_json')
def nodes_json():
    nodes_data = get_node_data()
    return jsonify(nodes_data)

@app.route('/nodes_count')
def count_filtered_elements():
    count = 0
    json_data = get_node_data()
    today_date_str = datetime.now().strftime('%Y-%m-%d')  # Get today's date in 'YYYY-MM-DD' format

    for element in json_data:
        # Check if timestamp contains today's date
        if 'timestamp' in element and today_date_str in element['timestamp']:
            # Check if hash has the specified format
            if 'hash' in element and element['hash'].count(':') == 2:
                count += 1
    return jsonify(count)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)
