#!/usr/bin/env python3
"""
Test script for visualization query functionality.
"""

import os
import sys
sys.path.insert(0, 'server')

from query_agent import QueryAgent
from organelle_db import OrganelleDatabase

# Setup paths
db_path = "./organelle_data/organelles.db"
csv_paths = os.getenv("ORGANELLE_CSV_PATHS", "").split(",")

print("[TEST] Initializing database...")
db = OrganelleDatabase(db_path=db_path, csv_paths=csv_paths)

print(f"[TEST] Database has {db.get_row_count()} organelles")
print(f"[TEST] Available types: {db.get_available_organelle_types()}")

# Initialize query agent
model = os.getenv("QUERY_AI_MODEL", "qwen2.5-coder:1.5b")
print(f"\n[TEST] Initializing QueryAgent with model: {model}")
agent = QueryAgent(db=db, model=model)

# Test queries
test_queries = [
    "show only the 3 largest mitos",
    "how many mitos are there?",
    "take me to the biggest nucleus",
]

print("\n" + "="*80)
for query in test_queries:
    print(f"\n[TEST] Query: {query}")
    print("-" * 80)

    result = agent.process_query(query)

    print(f"Type: {result.get('type')}")
    print(f"Answer: {result.get('answer')}")

    if result.get('sql'):
        print(f"SQL: {result.get('sql')}")

    if result.get('visualization'):
        viz = result['visualization']
        print(f"Visualization:")
        print(f"  Layer: {viz.get('layer_name')}")
        print(f"  Segments: {viz.get('segment_ids')[:5]}..." if len(viz.get('segment_ids', [])) > 5 else f"  Segments: {viz.get('segment_ids')}")
        print(f"  Action: {viz.get('action')}")

    if result.get('navigation'):
        nav = result['navigation']
        print(f"Navigation:")
        print(f"  Position: {nav.get('position')}")
        print(f"  Scale: {nav.get('scale')}")

    print("="*80)

print("\n[TEST] Done!")
