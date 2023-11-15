import sqlite3
from passlib.hash import argon2

# Function to generate Argon2 hash
def gen3erate_argon2_hash(key, m_value, t_value, p_value, salt_bytes):
    hash = argon2.using(rounds=t_value, memory_cost=m_value, parallelism=p_value, salt=salt_bytes, digest_size=64).hash(key)
    return hash


def generate_argon2_hash(key, m_value, t_value, p_value, salt_bytes):
    # Ensure key and salt_bytes are in bytes format
    if isinstance(key, str):
        key = key.encode()
    if isinstance(salt_bytes, str):
        salt_bytes = salt_bytes.encode()

    # Configure the Argon2 hasher
    argon2_hasher = argon2.using(time_cost=t_value, memory_cost=m_value, parallelism=p_value, salt=salt_bytes, hash_len=64)

    # Generate the hash using the configured hasher
    hashed_data = argon2_hasher.hash(key)

    return hashed_data


#0000da975bd6ec3aa878dadc395943619d23407371bc15066c1505ef23203d871633c687a9e5e89f5fc7fb61f05e1ff4ec49ecee28577c5143711185afe2d5a5
#select * from blocks where key = '0000da975bd6ec3aa878dadc395943619d23407371bc15066c1505ef23203d871633c687a9e5e89f5fc7fb61f05e1ff4ec49ecee28577c5143711185afe2d5a5';
#962524|$argon2id$v=19$m=1327,t=1,p=1$WEVOMTAwODIwMjJYRU4$Vn48cF2Es/CmZXEN11upLsyantNjLy7VP4h2v0z5WFBimDH7AE3kf6uP8AGe1umQncuthETqhKh8yJ7oCFUShg

# Example usage
key = "0000da975bd6ec3aa878dadc395943619d23407371bc15066c1505ef23203d871633c687a9e5e89f5fc7fb61f05e1ff4ec49ecee28577c5143711185afe2d5a5"          # Replace with your actual key
m_value = 1327 # Example m value
t_value = 1                    # Example t value
p_value = 1                    # Example p value
salt_bytes = b"XEN10082022XEN" # Replace with your actual salt bytes

argon2_hash = generate_argon2_hash(key, m_value, t_value, p_value, salt_bytes)
print("Argon2 Hash:", argon2_hash)

#exit(0)

# Connect to the SQLite database
conn = sqlite3.connect('balances6.db')  # Replace with your database file
cursor = conn.cursor()

# Query to retrieve rows from account_balances
cursor.execute("SELECT m_value, t_value, p_value, salt_bytes, key FROM account_balances")

# Process each row
for row in cursor.fetchall():
    m_value, t_value, p_value, salt_bytes, key = row

    #print ("key: ", key.hex())
    #print ("salt: ", salt_bytes)
    # Generate the Argon2 hash
    argon2_hash = generate_argon2_hash(key.hex(), m_value, t_value, p_value, salt_bytes)

    # Print or store the generated hash
    print("Generated Hash:", argon2_hash)

# Close the database connection
conn.close()

