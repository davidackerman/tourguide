# Quick Start: Visualization Queries

## How to Use

### 1. Start the Server

```bash
pixi run start
```

This starts the full Neuroglancer server with the query interface.

### 2. Open the Web UI

Navigate to the URL shown in the terminal (usually `http://localhost:8000`)

### 3. Try Visualization Queries

In the query box, try:

- **"show only the 3 largest mitos"** - Hides all mitochondria except the 3 biggest
- **"display the top 5 nuclei"** - Shows only the 5 largest nuclei
- **"highlight the smallest lysosome"** - Shows just one lysosome

### What Happens

1. ✅ Query is sent to the AI
2. ✅ AI generates SQL to find the organelles
3. ✅ **Neuroglancer viewer updates** to show only those segments
4. ✅ Audio narration plays describing what you're seeing

## Testing Without the Server

If you want to test the query logic without running Neuroglancer:

```bash
pixi run python debug_query.py "show only the 3 largest mitos"
```

This shows:
- ✅ What SQL is generated
- ✅ What segments are found
- ✅ What Python code **would** update Neuroglancer
- ❌ **Does NOT** actually update the viewer (no Neuroglancer running)

## Architecture

```
┌─────────────┐
│   Web UI    │ User types query
└──────┬──────┘
       │ HTTP POST /api/query/ask
       ↓
┌─────────────┐
│ Query Agent │ AI classifies intent → generates SQL
└──────┬──────┘
       │ Returns visualization command
       ↓
┌─────────────┐
│  stream.py  │ Updates Neuroglancer state
└──────┬──────┘
       │ with viewer.txn() as s:
       │     s.layers['mito_filled'].segments = {...}
       ↓
┌─────────────┐
│Neuroglancer │ Viewer updates instantly
└─────────────┘
```

## Where the Magic Happens

**Query Agent** ([query_agent.py](server/query_agent.py)):
- Classifies query intent using AI
- Generates SQL
- Extracts segment IDs and layer names

**State Update** ([stream.py:560-591](server/stream.py#L560-L591)):
- Receives visualization command
- Updates Neuroglancer layer segments
- Returns response to web UI

## Troubleshooting

**Q: Query runs but nothing happens in viewer?**
- Make sure you're using the **web UI**, not the debug script
- Check browser console for errors

**Q: Getting "Query contains unsafe operations"?**
- The AI model might have generated incomplete SQL
- Try rephrasing your query
- Consider using a larger model (nemotron-3-nano recommended)

**Q: Wrong segments are shown?**
- Check that the organelle type is spelled correctly in query
- Verify CSV data was imported correctly: check database organelle types

**Q: Want to reset view to show all segments?**
- Refresh the page, or
- Ask "show all mitos" (implementation needed for this)
