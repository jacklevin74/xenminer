import sqlite3

# Database connection
DATABASE_NAME = "blocks.db"
conn = sqlite3.connect(DATABASE_NAME)
cursor = conn.cursor()

cursor.execute("SELECT block_id FROM blocks WHERE block_id > 1004217 ORDER BY block_id ASC")
block_ids = [row[0] for row in cursor.fetchall()]

# Now, verify the sequence
is_sequence_correct = True
for i in range(1, len(block_ids)):
    if block_ids[i] - block_ids[i-1] != 1:
        print(f"Sequence broken between {block_ids[i-1]} and {block_ids[i]}")
        is_sequence_correct = False
        break

if is_sequence_correct:
    print("All block_ids are in sequence!")
else:
    print("block_ids are not in perfect sequence.")

conn.close()

