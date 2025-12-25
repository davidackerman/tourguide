"""
Organelle Database Layer

Manages SQLite database for organelle data with auto-import from CSV files.
"""

import sqlite3
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime


class OrganelleDatabase:
    """SQLite database for organelle data with CSV import capabilities."""

    def __init__(self, db_path: str, csv_paths: List[str]):
        """
        Initialize database and auto-import CSV files.

        Args:
            db_path: Path to SQLite database file
            csv_paths: List of CSV file paths to import
        """
        self.db_path = db_path
        self.csv_paths = [p for p in csv_paths if p.strip()]

        # Ensure database directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"[DB] Created directory: {db_dir}", flush=True)

        # Initialize database
        self._create_tables()

        # Import CSV files
        if self.csv_paths:
            self._import_all_csvs()
        else:
            print("[DB] No CSV paths provided, database empty", flush=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def _create_tables(self):
        """Create database schema if it doesn't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS organelles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id TEXT NOT NULL,
                organelle_type TEXT NOT NULL,
                volume REAL,
                surface_area REAL,
                position_x REAL,
                position_y REAL,
                position_z REAL,
                dataset_name TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_organelle_type
            ON organelles(organelle_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_volume
            ON organelles(volume)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_position
            ON organelles(position_x, position_y, position_z)
        """)

        conn.commit()
        conn.close()

        print(f"[DB] Database schema ready at {self.db_path}", flush=True)

    def _infer_organelle_type(self, csv_path: str) -> str:
        """
        Infer organelle type from CSV filename.

        Args:
            csv_path: Path to CSV file

        Returns:
            Organelle type (e.g., 'mitochondria', 'nucleus')
        """
        filename = Path(csv_path).stem.lower()

        # Common abbreviations to full names
        type_mapping = {
            'mito': 'mitochondria',
            'nuc': 'nucleus',
            'er': 'endoplasmic_reticulum',
            'golgi': 'golgi_apparatus',
            'ves': 'vesicle',
            'lyso': 'lysosome',
            'perox': 'peroxisome',
        }

        # Check if filename starts with any known abbreviation
        for abbrev, full_name in type_mapping.items():
            if filename.startswith(abbrev):
                return full_name

        # Otherwise use the filename as-is
        return filename

    def _import_csv(self, csv_path: str, organelle_type: str) -> int:
        """
        Import CSV file into database.

        Args:
            csv_path: Path to CSV file
            organelle_type: Type of organelle in this CSV

        Returns:
            Number of rows imported
        """
        if not os.path.exists(csv_path):
            print(f"[DB] CSV file not found: {csv_path}", flush=True)
            return 0

        try:
            # Read CSV with pandas
            df = pd.read_csv(csv_path)
            print(f"[DB] Loaded {len(df)} rows from {csv_path}", flush=True)

            # Map common column names to our schema
            column_mapping = {
                # Object ID variations
                'object_id': 'object_id',
                'id': 'object_id',
                'obj_id': 'object_id',
                'segment_id': 'object_id',

                # Volume variations
                'volume': 'volume',
                'vol': 'volume',
                'size': 'volume',

                # Surface area variations
                'surface_area': 'surface_area',
                'area': 'surface_area',
                'surf_area': 'surface_area',

                # Position variations
                'center_x': 'position_x',
                'x': 'position_x',
                'pos_x': 'position_x',
                'centroid_x': 'position_x',

                'center_y': 'position_y',
                'y': 'position_y',
                'pos_y': 'position_y',
                'centroid_y': 'position_y',

                'center_z': 'position_z',
                'z': 'position_z',
                'pos_z': 'position_z',
                'centroid_z': 'position_z',
            }

            # Rename columns (case-insensitive)
            df.columns = df.columns.str.lower()
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)

            # Prepare data for insertion
            conn = self._get_connection()
            cursor = conn.cursor()

            rows_inserted = 0
            for _, row in df.iterrows():
                # Extract known columns
                object_id = row.get('object_id', f"{organelle_type}_{rows_inserted}")
                volume = row.get('volume', None)
                surface_area = row.get('surface_area', None)
                position_x = row.get('position_x', None)
                position_y = row.get('position_y', None)
                position_z = row.get('position_z', None)

                # Store any additional columns in metadata JSON
                metadata = {}
                for col in df.columns:
                    if col not in ['object_id', 'volume', 'surface_area',
                                   'position_x', 'position_y', 'position_z']:
                        val = row.get(col)
                        if pd.notna(val):
                            metadata[col] = val

                metadata_json = json.dumps(metadata) if metadata else None

                # Insert into database
                cursor.execute("""
                    INSERT INTO organelles
                    (object_id, organelle_type, volume, surface_area,
                     position_x, position_y, position_z, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(object_id),
                    organelle_type,
                    float(volume) if pd.notna(volume) else None,
                    float(surface_area) if pd.notna(surface_area) else None,
                    float(position_x) if pd.notna(position_x) else None,
                    float(position_y) if pd.notna(position_y) else None,
                    float(position_z) if pd.notna(position_z) else None,
                    metadata_json
                ))

                rows_inserted += 1

            conn.commit()
            conn.close()

            print(f"[DB] Imported {rows_inserted} {organelle_type} records", flush=True)
            return rows_inserted

        except Exception as e:
            print(f"[DB] Error importing {csv_path}: {e}", flush=True)
            return 0

    def _import_all_csvs(self):
        """Import all CSV files specified in csv_paths."""
        total_imported = 0

        for csv_path in self.csv_paths:
            organelle_type = self._infer_organelle_type(csv_path)
            count = self._import_csv(csv_path, organelle_type)
            total_imported += count

        print(f"[DB] Total records imported: {total_imported}", flush=True)

    def execute_query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results as list of dictionaries.

        Args:
            sql: SQL query string
            params: Query parameters (for parameterized queries)

        Returns:
            List of result rows as dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

            # Convert to list of dicts
            results = []
            for row in rows:
                results.append(dict(row))

            conn.close()
            return results

        except Exception as e:
            conn.close()
            raise e

    def get_organelle_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Get organelle data by object ID.

        Args:
            object_id: Object ID to look up

        Returns:
            Organelle data dictionary or None if not found
        """
        results = self.execute_query(
            "SELECT * FROM organelles WHERE object_id = ?",
            (object_id,)
        )
        return results[0] if results else None

    def get_schema_description(self) -> str:
        """
        Get human-readable schema description for AI prompts.

        Returns:
            Schema description string
        """
        return """
Table: organelles

Columns:
- object_id (TEXT): Unique identifier for each organelle
- organelle_type (TEXT): Type of organelle (e.g., 'mitochondria', 'nucleus', 'endoplasmic_reticulum')
- volume (REAL): Volume in nm³ or voxels
- surface_area (REAL): Surface area
- position_x (REAL): X coordinate of center position
- position_y (REAL): Y coordinate of center position
- position_z (REAL): Z coordinate of center position
- metadata (TEXT): Additional properties as JSON

Available organelle types: {}
        """.format(", ".join(self.get_available_organelle_types()))

    def get_available_organelle_types(self) -> List[str]:
        """
        Get list of unique organelle types in database.

        Returns:
            List of organelle type strings
        """
        try:
            results = self.execute_query(
                "SELECT DISTINCT organelle_type FROM organelles ORDER BY organelle_type"
            )
            return [row['organelle_type'] for row in results]
        except:
            return []

    def get_row_count(self) -> int:
        """
        Get total number of organelles in database.

        Returns:
            Row count
        """
        try:
            results = self.execute_query("SELECT COUNT(*) as count FROM organelles")
            return results[0]['count'] if results else 0
        except:
            return 0
