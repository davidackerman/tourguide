#!/usr/bin/env python3
import sys
sys.path.insert(0, 'server')
from organelle_db import OrganelleDatabase

db = OrganelleDatabase('./organelle_data/organelles.db', [])

# Check actual object IDs in database
results = db.execute_query("""
    SELECT object_id, organelle_type, volume
    FROM organelles
    WHERE organelle_type='mitochondria'
    ORDER BY volume DESC
    LIMIT 5
""")

print("Top 5 mitochondria by volume:")
print("="*80)
for r in results:
    print(f"  ID: {r['object_id']:<20} Type: {r['organelle_type']:<20} Volume: {r['volume']:.2e}")
print()

# Check nuclei
results = db.execute_query("""
    SELECT object_id, organelle_type, volume
    FROM organelles
    WHERE organelle_type='nucleus'
    LIMIT 5
""")

print("First 5 nuclei:")
print("="*80)
for r in results:
    print(f"  ID: {r['object_id']:<20} Type: {r['organelle_type']:<20} Volume: {r['volume']:.2e}")
