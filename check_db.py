import sqlite3

db_path = './organelle_data/organelles.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get schema
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='organelles'")
schema = cursor.fetchone()
if schema:
    print("DATABASE SCHEMA:")
    print("=" * 80)
    print(schema[0])
    print("=" * 80)
    print()

# Get sample data
cursor.execute("SELECT * FROM organelles LIMIT 1")
row = cursor.fetchone()
if row:
    cursor.execute("PRAGMA table_info(organelles)")
    columns = cursor.fetchall()
    print("COLUMNS:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    print()

# Get count
cursor.execute("SELECT COUNT(*) FROM organelles")
count = cursor.fetchone()[0]
print(f"Total rows: {count}")

conn.close()
