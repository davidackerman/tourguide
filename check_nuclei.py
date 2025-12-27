import sqlite3

db_path = './organelle_data/organelles.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check all nuclei
print("=" * 80)
print("ALL NUCLEUS RECORDS:")
print("=" * 80)
cursor.execute("""
    SELECT id, object_id, volume, surface_area, position_x, position_y, position_z, dataset_name
    FROM organelles
    WHERE organelle_type = 'nucleus'
    ORDER BY volume DESC
""")

rows = cursor.fetchall()
print(f"Total nucleus records: {len(rows)}\n")

for i, row in enumerate(rows, 1):
    print(f"Row {i}:")
    print(f"  DB ID: {row[0]}")
    print(f"  Object ID: {row[1]}")
    print(f"  Volume: {row[2]}")
    print(f"  Surface Area: {row[3]}")
    print(f"  Position: ({row[4]}, {row[5]}, {row[6]})")
    print(f"  Dataset: {row[7]}")
    print()

# Check for duplicates
print("=" * 80)
print("DUPLICATE ANALYSIS:")
print("=" * 80)
cursor.execute("""
    SELECT object_id, COUNT(*) as count
    FROM organelles
    WHERE organelle_type = 'nucleus'
    GROUP BY object_id
    HAVING count > 1
""")

dupes = cursor.fetchall()
if dupes:
    print(f"Found {len(dupes)} object IDs with duplicates:\n")
    for obj_id, count in dupes:
        print(f"  {obj_id}: {count} copies")
else:
    print("No duplicates found")

conn.close()
