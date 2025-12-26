#!/usr/bin/env python3
"""
Comprehensive test for all query types.
"""

import os
import sys
sys.path.insert(0, 'server')

from query_agent import QueryAgent
from organelle_db import OrganelleDatabase

# Setup
db_path = "./organelle_data/organelles.db"
csv_paths = os.getenv("ORGANELLE_CSV_PATHS", "").split(",")

db = OrganelleDatabase(db_path=db_path, csv_paths=csv_paths)
model = os.getenv("QUERY_AI_MODEL", "qwen2.5-coder:1.5b")
agent = QueryAgent(db=db, model=model)

# Comprehensive test cases
test_cases = [
    ("show only the 5 biggest nuclei", "visualization"),
    ("display the top 10 mitochondria", "visualization"),
    ("show me only the smallest lysosome", "visualization"),
    ("how many peroxisomes are there?", "informational"),
    ("what is the average volume of nuclei?", "informational"),
    ("take me to the largest mito", "navigation"),
    ("go to the biggest cell", "navigation"),
]

print("="*80)
print("COMPREHENSIVE QUERY TYPE TEST")
print("="*80)

passed = 0
failed = 0

for query, expected_type in test_cases:
    result = agent.process_query(query)
    actual_type = result.get('type')

    status = "✓ PASS" if actual_type == expected_type else "✗ FAIL"
    if actual_type == expected_type:
        passed += 1
    else:
        failed += 1

    print(f"\n{status}")
    print(f"Query: {query}")
    print(f"Expected: {expected_type} | Actual: {actual_type}")

    if result.get('visualization'):
        viz = result['visualization']
        seg_count = len(viz.get('segment_ids', []))
        print(f"  → Showing {seg_count} segments in layer '{viz.get('layer_name')}'")

    if result.get('navigation'):
        nav = result['navigation']
        print(f"  → Navigating to position {nav.get('position')}")

print("\n" + "="*80)
print(f"Results: {passed} passed, {failed} failed")
print("="*80)
