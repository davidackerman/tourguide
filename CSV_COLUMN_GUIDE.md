# CSV Column Interpretation Guide

This guide explains how CSV columns are interpreted and mapped to the database.

## IMPORTANT: Units and Coordinate System

### Units
All measurements in the database use the following units:
- **Volume**: nm³ (cubic nanometers)
- **Surface Area**: nm² (square nanometers)
- **Position coordinates**: nm (nanometers)
- **Lengths/distances**: nm (nanometers) - e.g., 'lsp (nm)', 'radius mean (nm)'

**Note**: The CSV files specify "8nm" resolution in some column headers, but the actual coordinate values are already in nanometers, not voxels. The "8nm" refers to the voxel size of the imaging data.

### Coordinate System
- **Database columns**: `position_x, position_y, position_z` in (X, Y, Z) order, stored in **nanometers**
- **CSV columns**: Usually labeled as `COM X (nm), COM Y (nm), COM Z (nm)` in the same order, in **nanometers**
- **Neuroglancer state**: Uses (X, Y, Z) order in the `position` field, but in **voxel coordinates**

**IMPORTANT - Coordinate Conversion**:
- The database stores positions in **nanometers** (from the CSV files)
- Neuroglancer viewer uses **voxel coordinates**
- When navigating, the system automatically converts: `voxel_coords = nm_coords / voxel_size`
- For C. elegans dataset: voxel_size = (8, 8, 8) nm
- Example: Position (8000, 16000, 24000) nm → (1000, 2000, 3000) voxels

When the AI responds to queries, it now automatically includes proper units in all answers:
- "The volume is 3.05e11 nm³"
- "Taking you to mitochondria 123 at position (8000, 16000, 24000) nm" (shown in nm for clarity)
- "The average length is 456.7 nm"

The navigation command internally converts to voxel coordinates before updating the Neuroglancer viewer.

## Quick Inspection Tool

Use `inspect_csv.py` to see how any CSV file will be interpreted:

```bash
pixi run python inspect_csv.py /path/to/your_file.csv
```

This shows:
- All columns with their interpretations
- Sample values from each column
- Which columns map to database fields
- Which columns get stored as metadata

## Example: Mitochondria CSV

```
File: mito_filled_with_skeleton.csv
Rows: 1199 organelles
Columns: 16 fields
```

### Core Database Fields (automatically mapped)

| CSV Column | Database Field | Interpretation |
|------------|---------------|----------------|
| Object ID | object_id | Unique identifier for each organelle |
| Volume (nm^3) | volume | Volume in cubic nanometers |
| Surface Area (nm^2) | surface_area | Surface area in square nanometers |
| COM X (nm) | position_x | Center of mass X coordinate |
| COM Y (nm) | position_y | Center of mass Y coordinate |
| COM Z (nm) | position_z | Center of mass Z coordinate |

### Metadata Fields (stored as JSON)

These columns are preserved but stored in the `metadata` JSON field:

| Column | Interpretation |
|--------|----------------|
| MIN X/Y/Z (nm) | Bounding box minimum coordinates |
| MAX X/Y/Z (nm) | Bounding box maximum coordinates |
| lsp (nm) | Longest shortest path - morphological extent measure |
| radius mean (nm) | Average radius of the structure |
| radius std (nm) | Standard deviation of radius |
| num branches | Number of branches in skeleton |

## Column Name Patterns Recognized

The database automatically recognizes these patterns (case-insensitive):

### Object Identifiers
- `Object ID`, `object_id`, `id`, `obj_id`, `segment_id`

### Volume
- `Volume`, `vol`, `size`
- `Volume (nm^3)`, `Volume_(nm^3)`

### Surface Area
- `Surface Area`, `surface_area`, `area`, `surf_area`
- `Surface Area (nm^2)`, `Surface_Area_(nm^2)`

### Position - Center of Mass
- `COM X (nm)`, `com_x`, `center_x`, `centroid_x`, `x`, `pos_x`
- `COM Y (nm)`, `com_y`, `center_y`, `centroid_y`, `y`, `pos_y`
- `COM Z (nm)`, `com_z`, `center_z`, `centroid_z`, `z`, `pos_z`

### Morphological Features (stored as metadata)
- **Bounding Box**: `MIN X/Y/Z`, `MAX X/Y/Z`
- **Skeleton**: `lsp`, `num branches`, `num_branches`
- **Shape**: `radius mean`, `radius std`, `radius_mean`, `radius_std`
- **Descriptors**: `sphericity`, `aspect ratio`, `length`, `width`, `diameter`

## How Column Mapping Works

1. **CSV is loaded** with pandas
2. **Column names are normalized** to lowercase
3. **Pattern matching** finds known column types
4. **Core fields** are mapped to database columns:
   - object_id
   - volume
   - surface_area
   - position_x, position_y, position_z
5. **Extra fields** are stored as JSON in the `metadata` column
6. **Sample values** are logged during import

## Organelle Type Inference

The organelle type is inferred from the CSV filename:

| Filename Pattern | Organelle Type |
|-----------------|----------------|
| `mito_*.csv` | mitochondria |
| `nuc_*.csv` | nucleus |
| `er_*.csv` | endoplasmic_reticulum |
| `golgi_*.csv` | golgi_apparatus |
| `lyso_*.csv` | lysosome |
| `perox_*.csv` | peroxisome |
| `yolk_*.csv` | yolk_filled |
| `ld_*.csv` | lipid_droplet |
| `cell_*.csv` | cell |

## Adding New CSV Files

To add new organelle CSVs:

1. **Check column format** with the inspect tool:
   ```bash
   pixi run python inspect_csv.py /path/to/new_file.csv
   ```

2. **Add to .env file**:
   ```
   ORGANELLE_CSV_PATHS=/path/to/file1.csv,/path/to/file2.csv
   ```

3. **Delete old database** to reimport:
   ```bash
   rm organelle_data/organelles.db
   ```

4. **Restart server** - database will be recreated automatically

## Supported Column Types

The interpreter recognizes:

✓ **Identifiers**: Object IDs, segment IDs
✓ **Geometry**: Volume, surface area, positions
✓ **Coordinates**: COM (center of mass), centroids, positions
✓ **Bounding Boxes**: MIN/MAX coordinates
✓ **Morphology**: Skeleton features, radius, branches
✓ **Shape Descriptors**: Sphericity, aspect ratio
✓ **Measurements**: Length, width, diameter
✓ **Metadata**: Dataset names, timestamps

## Examples from Your Data

### C. elegans Dataset Files

All files in: `/nrs/cellmap/ackermand/cellmap/analysisResults/c-elegans/jrc_c-elegans-op50-1/`

| File | Organelles | Special Columns |
|------|-----------|----------------|
| `mito_filled_with_skeleton.csv` | 1,199 | lsp, radius mean/std, branches |
| `lyso_filled.csv` | 3,200 | Standard fields only |
| `perox_filled.csv` | 4,367 | Standard fields only |
| `nuc_filled.csv` | 4 | Standard fields only |
| `yolk_filled.csv` | 44 | Standard fields only |
| `cell.csv` | 7 | Standard fields only |

## Querying Metadata

Metadata fields are stored as JSON and can be queried:

```sql
-- Extract metadata field
SELECT object_id,
       json_extract(metadata, '$.lsp (nm)') as longest_path
FROM organelles
WHERE organelle_type = 'mitochondria';

-- Find mitos with many branches
SELECT object_id,
       json_extract(metadata, '$.num branches') as branches
FROM organelles
WHERE organelle_type = 'mitochondria'
  AND json_extract(metadata, '$.num branches') > 5;
```

## Troubleshooting

**Position data is NULL**:
- Check that your CSV has `COM X/Y/Z` or similar columns
- Run `inspect_csv.py` to see what's being mapped

**Wrong organelle type**:
- Rename CSV file to match pattern (e.g., `mito_*.csv`)
- Or manually specify in database code

**Missing columns**:
- Use `inspect_csv.py` to see interpretations
- Add new patterns to column_mapping in `organelle_db.py`

**Metadata not showing**:
- Check that column wasn't mapped to a core field
- Use JSON functions in SQL to access metadata
