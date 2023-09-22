import sqlite3

def get_distinct_eth_addresses_with_same_last_block_id(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Execute the SQL query
    query = '''
    SELECT my_ethereum_address, last_block_id, COUNT(id) AS m
    FROM consensus
    GROUP BY my_ethereum_address, last_block_id
    HAVING m > 1
    '''
    
    cursor.execute(query)
    
    # Fetch all the records
    records = cursor.fetchall()
    
    # Close the database connection
    conn.close()
    
    # Create a set to hold distinct Ethereum addresses
    distinct_eth_addresses = set()
    
    # Extract distinct Ethereum addresses from the records
    for record in records:
        distinct_eth_addresses.add(record[0])
    
    # Report the number of distinct Ethereum addresses
    print(f"Number of distinct my_ethereum_address records with the same last_block_id: {len(distinct_eth_addresses)}")

# Path to your SQLite database
db_path = 'blocks.db'

# Run the function
get_distinct_eth_addresses_with_same_last_block_id(db_path)

