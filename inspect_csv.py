#!/usr/bin/env python3
"""
Utility to inspect and interpret CSV column names.
Shows what each column means and how it will be mapped to the database.

Usage:
    python inspect_csv.py <csv_file>
    python inspect_csv.py /path/to/mito_filled_with_skeleton.csv
"""

import sys
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from organelle_db import OrganelleDatabase
import pandas as pd


def print_separator(char="-", length=100):
    print(char * length)


def inspect_csv(csv_path: str):
    """Inspect a CSV file and show column interpretations."""

    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        return

    print("=" * 100)
    print(f"CSV FILE: {csv_path}")
    print("=" * 100)
    print()

    # Get interpretations
    try:
        interpretations = OrganelleDatabase.interpret_csv_columns(csv_path)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return

    # Read sample data
    df = pd.read_csv(csv_path, nrows=3)

    print(f"ROWS IN FILE: {len(pd.read_csv(csv_path))}")
    print(f"COLUMNS: {len(df.columns)}")
    print()

    print_separator("=")
    print("COLUMN INTERPRETATIONS")
    print_separator("=")
    print()

    # Print each column with its interpretation
    max_col_len = max(len(col) for col in df.columns)

    for i, col in enumerate(df.columns, 1):
        interpretation = interpretations.get(col, "Unknown")

        print(f"{i:2d}. {col:<{max_col_len}}  →  {interpretation}")

        # Show sample values
        sample_values = df[col].head(3).tolist()
        if any(pd.notna(v) for v in sample_values):
            print(f"    Samples: {sample_values}")
        print()

    print_separator("=")
    print("DATABASE MAPPING")
    print_separator("=")
    print()

    # Show what will be stored in the main database columns vs metadata
    col_lower_list = [c.lower() for c in df.columns]

    # These are the patterns we look for
    mapping_checks = {
        'object_id': ['object id', 'object_id', 'id'],
        'volume': ['volume (nm^3)', 'volume_(nm^3)', 'volume', 'vol'],
        'surface_area': ['surface area (nm^2)', 'surface_area_(nm^2)', 'surface_area', 'area'],
        'position_x': ['com x (nm)', 'com_x_(nm)', 'com_x', 'center_x', 'x'],
        'position_y': ['com y (nm)', 'com_y_(nm)', 'com_y', 'center_y', 'y'],
        'position_z': ['com z (nm)', 'com_z_(nm)', 'com_z', 'center_z', 'z'],
    }

    mapped_columns = []
    for db_col, patterns in mapping_checks.items():
        for pattern in patterns:
            if pattern in col_lower_list:
                original_col = df.columns[col_lower_list.index(pattern)]
                print(f"✓ {db_col:<15} ← {original_col}")
                mapped_columns.append(original_col)
                break

    print()
    print("METADATA (stored as JSON):")
    for col in df.columns:
        if col not in mapped_columns:
            print(f"  • {col}")

    print()
    print_separator("=")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_csv.py <csv_file>")
        print()
        print("Example:")
        print("  python inspect_csv.py /path/to/mito_filled_with_skeleton.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    inspect_csv(csv_path)


if __name__ == "__main__":
    main()
