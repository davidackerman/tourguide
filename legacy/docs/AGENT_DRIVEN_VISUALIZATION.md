# Agent-Driven Visualization System

## Overview

The visualization system has been refactored to use AI-driven decision making instead of hardcoded logic. The agent now receives the user query, SQL results, and current Neuroglancer state, and generates appropriate visualization commands.

## What Changed

### Before (Hardcoded Approach)

**File**: [`query_agent.py:1329-1401`](server/query_agent.py#L1329-L1401) (old version)

The old `_handle_visualization_query` method:
1. ❌ Hardcoded extraction of segment IDs
2. ❌ Hardcoded layer name mapping
3. ❌ Hardcoded action (`show_only`)
4. ❌ Separate AI call just for formatting the answer

**Limitations**:
- Could not distinguish "show X" vs "also show X" vs "hide X"
- Required code changes for new visualization patterns
- Layer mapping logic duplicated between AI and hardcoded fallback
- No awareness of current viewer state

### After (Agent-Driven Approach)

**File**: [`query_agent.py:1329-1597`](server/query_agent.py#L1329-L1597)

The new system:
1. ✅ **AI decides everything** via `_generate_visualization_state_update()`
2. ✅ Receives current NG state as context
3. ✅ Generates visualization commands AND answer in one call
4. ✅ Interprets query semantics ("show", "also show", "hide")
5. ✅ Fallback to simple logic if AI fails

## New Methods

### `_generate_visualization_state_update()`

**Location**: [`query_agent.py:1379-1534`](server/query_agent.py#L1379-L1534)

**Purpose**: AI-powered visualization state update generator

**Inputs**:
- `user_query`: Original user question
- `sql_results`: Results from database query
- `ai_interactions`: Optional tracking list

**Outputs**:
```python
{
    "commands": [
        {
            "layer_name": "mito",
            "segment_ids": ["mito_123", "mito_456"],
            "action": "show_only"  # or "add" or "remove"
        }
    ],
    "answer": "Showing the 2 largest mitochondria..."
}
```

**AI Prompt Features**:
- Receives current Neuroglancer state (visible layers, selected segments)
- Receives available layer names from NG
- Receives SQL query results
- Given semantic guidelines for interpreting queries:
  - "show X" → `show_only` (replace current selection)
  - "also show X" → `add` (add to current selection)
  - "hide X" → `remove` (remove from selection)
- Outputs structured JSON with commands + natural language answer
- Requests JSON format from Ollama for reliable parsing

### `_fallback_visualization_update()`

**Location**: [`query_agent.py:1536-1597`](server/query_agent.py#L1536-L1597)

**Purpose**: Fallback when AI fails or is unavailable

**Logic**: Simple hardcoded approach similar to old system
- Extracts segment IDs from results
- Maps organelle type to layer using `_get_layer_name_fallback()`
- Always uses `show_only` action
- Generates simple answer text

## Benefits

### 1. **Flexibility**
The agent can handle nuanced queries:
- "show largest mito" → shows only that mito
- "also show nucleus 5" → adds nucleus 5 to current selection
- "hide all mitochondria" → removes all mitos from view
- "compare mito 1 and mito 2" → shows both side-by-side

### 2. **Context Awareness**
Agent sees current state and can make intelligent decisions:
- Knows what's already visible
- Can preserve or replace selections appropriately
- Could potentially adjust other properties (opacity, colors) in future

### 3. **Extensibility**
New visualization patterns require no code changes:
- Multi-layer visualizations
- Complex filtering
- Comparative views
- Custom display modes

### 4. **Consistency**
Single AI call handles both:
- Visualization command generation
- Natural language answer formatting

### 5. **Maintainability**
- Less hardcoded logic to maintain
- Easier to extend with new capabilities
- Clearer separation of concerns

## Architecture Flow

```
User Query: "show the largest mitochondrion"
           ↓
   process_query()
           ↓
   Classify intent: "visualization"
           ↓
   Generate SQL: "SELECT * FROM organelles WHERE organelle_type='mitochondria' ORDER BY volume DESC LIMIT 1"
           ↓
   Execute SQL → Results: [{object_id: 123, organelle_type: "mitochondria", volume: 3.5e11, ...}]
           ↓
   _handle_visualization_query()
           ↓
   _generate_visualization_state_update()
           ↓
   Get current NG state: {layers: [...], ...}
           ↓
   AI Prompt with:
   - User query
   - SQL results
   - Current NG state
   - Available layers
           ↓
   AI Response (JSON):
   {
     "commands": [{
       "layer_name": "mito",
       "segment_ids": ["mitochondria_123"],
       "action": "show_only"
     }],
     "answer": "Showing mitochondrion 123 with volume 3.50e+11 nm³"
   }
           ↓
   Return to frontend → Update Neuroglancer viewer
```

## Testing

**Test File**: [`test_viz_simple.py`](test_viz_simple.py)

Tests verify:
- ✅ New methods exist
- ✅ Fallback logic works correctly
- ✅ Single object visualization
- ✅ Multiple object visualization
- ✅ Empty results handling

## Future Enhancements

Potential future improvements enabled by this architecture:

1. **Smart layer visibility**
   - "show only mitochondria" could hide all other layers
   - "compare X and Y" could show multiple layers

2. **Property adjustments**
   - "highlight X in red"
   - "make Y semi-transparent"
   - "dim everything except X"

3. **Camera positioning**
   - "center on X"
   - "zoom to fit Y"
   - "show X and Y side by side"

4. **Multi-step interactions**
   - Agent remembers previous state
   - "now show the nucleus too" adds to current view

All of these can be implemented by extending the AI prompt without changing the core logic.

## Backward Compatibility

- ✅ Fallback system ensures queries work even if AI fails
- ✅ Old helper methods (`_get_layer_name`, `_format_visualization_answer`) still exist but deprecated
- ✅ Frontend integration unchanged (still receives same command structure)
- ✅ Can gradually migrate to more sophisticated patterns

## Performance

- Single AI call instead of multiple (was: intent + SQL + layer mapping + answer formatting)
- JSON format request ensures reliable parsing
- Fallback activates quickly if AI fails
- Current state extracted once per query

## Configuration

No configuration changes needed. The system automatically:
- Detects available Neuroglancer layers at init
- Adapts to available organelle types
- Works with or without ng_tracker
