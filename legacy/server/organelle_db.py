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
import ollama
from ollama_manager import ensure_ollama_running


class OrganelleDatabase:
    """SQLite database for organelle data with CSV import capabilities."""

    def __init__(self, db_path: str, csv_paths: List[str], ai_model: Optional[str] = None):
        """
        Initialize database and auto-import CSV files.

        Args:
            db_path: Path to SQLite database file
            csv_paths: List of CSV file paths to import
            ai_model: Optional AI model for intelligent column/type inference
        """
        self.db_path = db_path
        self.csv_paths = [p for p in csv_paths if p.strip()]
        self.ai_model = ai_model  # For AI-driven column mapping and type inference

        # Ensure Ollama is running if AI model is specified
        if self.ai_model:
            if not ensure_ollama_running():
                print("[DB] Warning: Failed to start Ollama, AI features may not work", flush=True)

        # Ensure database directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"[DB] Created directory: {db_dir}", flush=True)

        # Initialize database
        self._create_tables()

        # Cache metadata schema for AI prompts
        self._metadata_schema = None

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
        Infer organelle type from CSV filename using AI or fallback.

        Args:
            csv_path: Path to CSV file

        Returns:
            Organelle type (e.g., 'mitochondria', 'nucleus')
        """
        # Try AI-based inference if model is available
        if self.ai_model:
            try:
                return self._infer_organelle_type_ai(csv_path)
            except Exception as e:
                print(f"[DB] AI inference failed: {e}, using fallback", flush=True)

        # Fallback to pattern matching
        return self._infer_organelle_type_fallback(csv_path)

    def _infer_organelle_type_ai(self, csv_path: str) -> str:
        """
        Use AI to infer organelle type from CSV filename and structure.

        Args:
            csv_path: Path to CSV file

        Returns:
            Organelle type inferred by AI
        """
        filename = Path(csv_path).stem
        df = pd.read_csv(csv_path, nrows=3)
        columns = list(df.columns)

        prompt = f"""Determine the organelle type from this CSV file.

Filename: {filename}
Columns: {columns}

Common organelle types: mitochondria, nucleus, endoplasmic_reticulum,
golgi_apparatus, lysosome, peroxisome, vesicle, cell, yolk, lipid_droplet

Provide ONLY the organelle type name (lowercase, underscore_separated).
If uncertain, normalize the filename (e.g., "mito_filled" → "mitochondria").

Organelle type:"""

        response = ollama.chat(model=self.ai_model, messages=[{"role": "user", "content": prompt}])
        organelle_type = response["message"]["content"].strip().lower()

        print(f"[DB] AI inferred organelle type '{organelle_type}' from {filename}", flush=True)
        return organelle_type

    def _infer_organelle_type_fallback(self, csv_path: str) -> str:
        """
        Fallback: Infer organelle type from CSV filename using pattern matching.

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
            'ld': 'lipid_droplet',
            'cell': 'cell',
            'yolk': 'yolk',
        }

        # Check if filename starts with any known abbreviation
        for abbrev, full_name in type_mapping.items():
            if filename.startswith(abbrev):
                return full_name

        # Otherwise use the filename as-is
        return filename

    def _analyze_columns_ai(self, csv_path: str) -> Dict[str, str]:
        """
        Use AI to map CSV columns to database schema fields.

        Args:
            csv_path: Path to CSV file

        Returns:
            Dict mapping CSV column names to database field names
        """
        df = pd.read_csv(csv_path, nrows=3)
        columns = list(df.columns)
        sample_data = df.head(2).to_dict('records')

        prompt = f"""Analyze this CSV and map columns to our database schema.

CSV Columns: {columns}
Sample Data (first 2 rows): {sample_data}

Database schema fields:
- object_id: Unique identifier for each organelle (required)
- volume: Volume measurement (prefer nm^3 units)
- surface_area: Surface area measurement (prefer nm^2 units)
- position_x, position_y, position_z: Center coordinates (prefer COM/centroid in nm)

For each CSV column that matches a database field, provide the mapping.
Return ONLY as JSON: {{"CSV Column Name": "database_field"}}

Examples:
- "Object ID" → "object_id"
- "Volume (nm^3)" → "volume"
- "COM X (nm)" → "position_x"

Columns that don't match (like MIN X, MAX X, lsp, etc.) should be OMITTED.

JSON mapping:"""

        try:
            response = ollama.chat(model=self.ai_model, messages=[{"role": "user", "content": prompt}])
            mapping_str = response["message"]["content"].strip()

            # Remove markdown code blocks if present
            if '```' in mapping_str:
                mapping_str = mapping_str.split('```')[1]
                if mapping_str.startswith('json'):
                    mapping_str = mapping_str[4:]
                mapping_str = mapping_str.strip()

            mapping = json.loads(mapping_str)
            print(f"[DB] AI column mapping for {Path(csv_path).name}: {mapping}", flush=True)
            return mapping

        except Exception as e:
            print(f"[DB] AI column mapping failed: {e}, using fallback", flush=True)
            return self._get_column_mapping_fallback()

    def _get_column_mapping_fallback(self) -> Dict[str, str]:
        """Fallback hardcoded column mapping."""
        return {
            # Object ID variations
            'object id': 'object_id',
            'object_id': 'object_id',
            'id': 'object_id',
            'obj_id': 'object_id',
            'segment_id': 'object_id',

            # Volume variations
            'volume': 'volume',
            'vol': 'volume',
            'size': 'volume',
            'volume_(nm^3)': 'volume',
            'volume (nm^3)': 'volume',

            # Surface area variations
            'surface_area': 'surface_area',
            'area': 'surface_area',
            'surf_area': 'surface_area',
            'surface_area_(nm^2)': 'surface_area',
            'surface area (nm^2)': 'surface_area',

            # Position variations
            'center_x': 'position_x',
            'x': 'position_x',
            'pos_x': 'position_x',
            'centroid_x': 'position_x',
            'com_x': 'position_x',
            'com_x_(nm)': 'position_x',
            'com x (nm)': 'position_x',

            'center_y': 'position_y',
            'y': 'position_y',
            'pos_y': 'position_y',
            'centroid_y': 'position_y',
            'com_y': 'position_y',
            'com_y_(nm)': 'position_y',
            'com y (nm)': 'position_y',

            'center_z': 'position_z',
            'z': 'position_z',
            'pos_z': 'position_z',
            'centroid_z': 'position_z',
            'com_z': 'position_z',
            'com_z_(nm)': 'position_z',
            'com z (nm)': 'position_z',
        }

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

            # Get column mapping (AI-based if model available, otherwise fallback)
            if self.ai_model:
                try:
                    column_mapping = self._analyze_columns_ai(csv_path)
                except Exception as e:
                    print(f"[DB] AI column mapping failed: {e}, using fallback", flush=True)
                    column_mapping = self._get_column_mapping_fallback()
            else:
                column_mapping = self._get_column_mapping_fallback()

            # Log original columns for debugging
            print(f"[DB] Original columns: {list(df.columns)}", flush=True)

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
                object_id_raw = row.get('object_id', f"{organelle_type}_{rows_inserted}")

                # Convert object_id to int if it's numeric (remove .0 from pandas floats)
                if pd.notna(object_id_raw):
                    try:
                        object_id = str(int(float(object_id_raw)))
                    except (ValueError, TypeError):
                        object_id = str(object_id_raw)
                else:
                    object_id = f"{organelle_type}_{rows_inserted}"

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
                    object_id,  # Already converted to string (int format)
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
        # Check if data already exists to prevent duplicate imports
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM organelles")
        existing_count = cursor.fetchone()[0]
        conn.close()

        if existing_count > 0:
            print(f"[DB] Database already contains {existing_count} records, skipping import", flush=True)
            return

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
        Get comprehensive schema description including metadata fields for AI prompts.

        Returns:
            Detailed schema description string
        """
        available_types = self.get_available_organelle_types()
        metadata_schema = self.get_metadata_schema()

        desc = """
Table: organelles

Core Columns (directly accessible):
- object_id (TEXT): Unique identifier for each organelle
- organelle_type (TEXT): Type of organelle
- volume (REAL): Volume in nm³
- surface_area (REAL): Surface area in nm²
- position_x (REAL): X coordinate of center position (nm)
- position_y (REAL): Y coordinate of center position (nm)
- position_z (REAL): Z coordinate of center position (nm)
- metadata (TEXT): Additional properties as JSON (see below)

Available organelle types: {}

Metadata Fields (access via json_extract):
""".format(", ".join(available_types))

        # Add metadata fields per organelle type
        for org_type in sorted(metadata_schema.keys()):
            fields = metadata_schema[org_type]
            if fields:
                desc += f"\n{org_type}:\n"
                for field in fields:
                    desc += f"  - {field}: json_extract(metadata, '$.{field}')\n"
            else:
                desc += f"\n{org_type}:\n  - (no additional metadata)\n"

        return desc

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

    def get_metadata_schema(self, force_refresh: bool = False) -> Dict[str, List[str]]:
        """
        Get metadata fields available for each organelle type with caching.

        Args:
            force_refresh: Force re-discovery of schema

        Returns:
            Dict mapping organelle_type -> list of metadata field names
        """
        if self._metadata_schema is None or force_refresh:
            self._metadata_schema = self._discover_metadata_schema()
        return self._metadata_schema

    def _discover_metadata_schema(self) -> Dict[str, List[str]]:
        """
        Discover metadata fields by sampling database.

        Returns:
            Dict mapping organelle_type -> list of metadata field names
        """
        schema = {}

        try:
            organelle_types = self.get_available_organelle_types()

            for org_type in organelle_types:
                # Sample first record with non-null metadata
                results = self.execute_query(
                    "SELECT metadata FROM organelles WHERE organelle_type = ? AND metadata IS NOT NULL LIMIT 1",
                    (org_type,)
                )

                if results and results[0].get('metadata'):
                    metadata = json.loads(results[0]['metadata'])
                    schema[org_type] = list(metadata.keys())
                else:
                    schema[org_type] = []

            return schema
        except Exception as e:
            print(f"[DB] Error discovering metadata schema: {e}", flush=True)
            return {}

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

    @staticmethod
    def interpret_csv_columns(csv_path: str) -> Dict[str, str]:
        """
        Analyze a CSV file and interpret what each column means.

        Args:
            csv_path: Path to CSV file

        Returns:
            Dictionary mapping column names to their interpretations
        """
        import pandas as pd

        # Read just the header and first row
        df = pd.read_csv(csv_path, nrows=1)

        interpretations = {}

        for col in df.columns:
            col_lower = col.lower()

            # Object ID
            if any(x in col_lower for x in ['object', 'id', 'segment']):
                interpretations[col] = "Unique identifier for each object/organelle"

            # Volume
            elif 'volume' in col_lower or 'vol' in col_lower:
                if 'nm^3' in col or 'nm³' in col:
                    interpretations[col] = "Volume in cubic nanometers (nm³)"
                else:
                    interpretations[col] = "Volume/size measurement"

            # Surface Area
            elif 'surface' in col_lower or 'area' in col_lower:
                if 'nm^2' in col or 'nm²' in col:
                    interpretations[col] = "Surface area in square nanometers (nm²)"
                else:
                    interpretations[col] = "Surface area measurement"

            # Center of Mass (COM)
            elif 'com x' in col_lower or 'com_x' in col_lower:
                interpretations[col] = "Center of mass X coordinate (nanometers)"
            elif 'com y' in col_lower or 'com_y' in col_lower:
                interpretations[col] = "Center of mass Y coordinate (nanometers)"
            elif 'com z' in col_lower or 'com_z' in col_lower:
                interpretations[col] = "Center of mass Z coordinate (nanometers)"

            # Bounding box MIN coordinates
            elif 'min x' in col_lower or 'min_x' in col_lower:
                interpretations[col] = "Minimum X coordinate of bounding box (nanometers)"
            elif 'min y' in col_lower or 'min_y' in col_lower:
                interpretations[col] = "Minimum Y coordinate of bounding box (nanometers)"
            elif 'min z' in col_lower or 'min_z' in col_lower:
                interpretations[col] = "Minimum Z coordinate of bounding box (nanometers)"

            # Bounding box MAX coordinates
            elif 'max x' in col_lower or 'max_x' in col_lower:
                interpretations[col] = "Maximum X coordinate of bounding box (nanometers)"
            elif 'max y' in col_lower or 'max_y' in col_lower:
                interpretations[col] = "Maximum Y coordinate of bounding box (nanometers)"
            elif 'max z' in col_lower or 'max_z' in col_lower:
                interpretations[col] = "Maximum Z coordinate of bounding box (nanometers)"

            # Generic position/coordinates
            elif any(x in col_lower for x in ['center', 'centroid', 'position', 'pos']):
                if 'x' in col_lower:
                    interpretations[col] = "X coordinate/position"
                elif 'y' in col_lower:
                    interpretations[col] = "Y coordinate/position"
                elif 'z' in col_lower:
                    interpretations[col] = "Z coordinate/position"
                else:
                    interpretations[col] = "Position/coordinate data"

            # Morphology features
            elif 'lsp' in col_lower:
                interpretations[col] = "Longest shortest path - morphological measure of extent (nanometers)"
            elif 'radius' in col_lower:
                if 'mean' in col_lower or 'avg' in col_lower:
                    interpretations[col] = "Average radius measurement (nanometers)"
                elif 'std' in col_lower or 'stdev' in col_lower:
                    interpretations[col] = "Standard deviation of radius measurements (nanometers)"
                else:
                    interpretations[col] = "Radius measurement (nanometers)"
            elif 'branch' in col_lower:
                interpretations[col] = "Number of branches in skeleton/structure"
            elif 'length' in col_lower:
                interpretations[col] = "Length measurement"
            elif 'width' in col_lower or 'diameter' in col_lower:
                interpretations[col] = "Width/diameter measurement"
            elif 'sphericity' in col_lower or 'roundness' in col_lower:
                interpretations[col] = "Shape descriptor (0-1, how sphere-like)"
            elif 'aspect' in col_lower:
                interpretations[col] = "Aspect ratio measurement"

            # Metadata
            elif 'dataset' in col_lower:
                interpretations[col] = "Dataset name/identifier"
            elif 'created' in col_lower or 'timestamp' in col_lower:
                interpretations[col] = "Timestamp/creation date"

            # Unknown
            else:
                # Try to infer from sample value
                sample_val = df[col].iloc[0]
                if pd.notna(sample_val):
                    if isinstance(sample_val, (int, float)):
                        interpretations[col] = f"Numeric measurement (sample: {sample_val})"
                    else:
                        interpretations[col] = f"Data field (sample: {sample_val})"
                else:
                    interpretations[col] = "Unknown data field"

        return interpretations
