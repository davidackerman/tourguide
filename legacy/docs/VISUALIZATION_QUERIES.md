# Visualization Query System

## Overview

The query agent now supports three types of queries:

1. **Informational** - Get statistics without changing the view
2. **Navigation** - Move the camera to view a specific organelle
3. **Visualization** - Show/hide specific segments in the viewer ⭐ NEW

## How It Works

### AI-Driven Intent Classification

Instead of using hardcoded keywords, the system uses the AI model to classify the user's intent:

```python
User Query: "show only the 3 largest mitos"
  ↓
AI Classification: "visualization"
  ↓
Generate SQL: SELECT object_id, organelle_type, volume
              FROM organelles
              WHERE organelle_type='mitochondria'
              ORDER BY volume DESC LIMIT 3;
  ↓
Extract Results: ['mitochondria_371', 'mitochondria_166', 'mitochondria_160']
  ↓
Update Neuroglancer: layer.segments = {'mitochondria_371', 'mitochondria_166', 'mitochondria_160'}
```

### Components Modified

1. **`query_agent.py`**
   - `_classify_query_type()` - AI-based intent classification
   - `_handle_visualization_query()` - Extract segment IDs and create visualization command
   - `_get_layer_name()` - Map organelle types to Neuroglancer layer names
   - `_format_visualization_answer()` - Generate natural language response

2. **`stream.py`**
   - Added visualization handler in `/api/query/ask` endpoint
   - Updates Neuroglancer state with segment visibility

## Example Queries

### Visualization Queries
- "show only the 3 largest mitos"
- "display the top 5 nuclei"
- "highlight mitochondria with volume > 10000"
- "show me the smallest ER"

### Navigation Queries
- "take me to the biggest mito"
- "go to nucleus 5"
- "navigate to the largest lysosome"

### Informational Queries
- "how many mitos are there?"
- "what is the average volume of nuclei?"
- "describe the largest ER"

## Visualization Actions

The system supports three types of visualization actions:

1. **show_only** (default) - Hide all segments except the specified ones
2. **add** - Add segments to the currently visible set (not yet exposed to users)
3. **remove** - Remove segments from the currently visible set (not yet exposed to users)

## Layer Mapping

Organelle types are automatically mapped to Neuroglancer layers:

| Database Type | Layer Name |
|--------------|------------|
| mitochondria | mito_filled |
| nucleus | nuc |
| endoplasmic_reticulum | er |
| lysosome | lyso |
| peroxisome | perox |
| cell | cell |
| yolk_filled | yolk |

## Technical Details

### Query Flow

```
User Query: "show only the 3 largest mitos"
    ↓
[Web UI] → POST /api/query/ask
    ↓
[AI] Classify Intent → "visualization"
    ↓
[AI] Generate SQL → SELECT object_id, organelle_type, volume
                    FROM organelles
                    WHERE organelle_type='mitochondria'
                    ORDER BY volume DESC LIMIT 3;
    ↓
[Database] Execute → Returns: mitochondria_371, mitochondria_166, mitochondria_160
    ↓
[QueryAgent] Extract segment IDs & layer name
    ↓
[Neuroglancer Update]
    with viewer.txn() as s:
        s.layers['mito_filled'].segments = {'mitochondria_371', 'mitochondria_166', 'mitochondria_160'}
    ↓
[Web UI] Shows updated view + plays audio narration
```

**Key Point**: The Neuroglancer state update happens in [stream.py:560-591](server/stream.py#L560-L591) when you run the full server (`pixi run start`). The debug script only shows what *would* be sent to Neuroglancer.

### State Management

Neuroglancer state is updated using transactions:

```python
with viewer.txn() as s:
    layer = s.layers["mito_filled"]
    layer.segments = {"mitochondria_371", "mitochondria_166", "mitochondria_160"}
```

This ensures atomic updates and proper state synchronization.

## Testing

Run the test script to verify functionality:

```bash
pixi run python test_visualization.py
```

Expected output shows:
- ✅ Correct intent classification
- ✅ Valid SQL generation
- ✅ Proper segment ID extraction
- ✅ Correct layer mapping

### Model Recommendations

The system works best with larger, more capable models:

- **Recommended**: `nemotron-3-nano` (24GB) - Best SQL generation and intent classification
- **Minimum**: `qwen2.5-coder:1.5b` (1GB) - Works but may have occasional SQL errors
- **Not recommended**: `qwen2.5:0.5b` (400MB) - Too small, generates incomplete SQL

To change models, edit `.env`:
```bash
QUERY_AI_MODEL=nemotron-3-nano
```

### Smart Fallbacks

The system includes intelligent fallbacks to handle model confusion:

1. **Navigation vs Visualization**: If a query returns a single result with position data, it's auto-corrected to navigation
2. **Error Handling**: Incomplete SQL is caught and a helpful error message is returned

## Future Enhancements

Possible improvements:
- Support for "add" and "remove" actions via natural language
- Combined queries (e.g., "show the 3 largest mitos AND navigate to the biggest one")
- Filtering by multiple criteria (e.g., "show mitos with volume between X and Y")
- Color coding based on properties (e.g., "color mitos by volume")
