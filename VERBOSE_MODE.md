# Verbose Mode - AI Interaction Logging

## Overview

Verbose mode shows all AI prompts and responses during query processing, making it easy to debug and understand how the system works.

## Usage

Add `--verbose` or `-v` flag to debug_query.py:

```bash
pixi run python debug_query.py --verbose "show me the 3 largest mitos"
pixi run python debug_query.py -v "your query here"
```

## What You See

### 1. Intent Classification

Shows the prompt sent to classify the query intent:

```
================================================================================
[AI] INTENT CLASSIFICATION
================================================================================
PROMPT:
Classify the intent of this user query about organelle data.

User Query: show me the 3 largest mitos

Intent Types:
1. navigation - User wants to go to/view a specific location
2. visualization - User wants to show/hide/filter specific objects
3. informational - User wants statistical information

Respond with ONLY one word: navigation, visualization, or informational

Intent:
--------------------------------------------------------------------------------
RESPONSE:
visualization
================================================================================
```

### 2. SQL Generation

Shows the full prompt with schema, fuzzy matching rules, and examples:

```
================================================================================
[AI] SQL GENERATION
================================================================================
PROMPT:
You are an expert at converting natural language to SQLite queries.

Database Schema:
[full schema shown]

IMPORTANT: The user may use informal names for organelles.
Available organelle types in database: cell, lysosome, mitochondria, ...

Examples of fuzzy name matching:
- "mitos" or "mito" → use 'mitochondria'
- "nuclei" or "nuc" → use 'nucleus'
...

User Query: show me the 3 largest mitos

[full prompt with rules and examples]
--------------------------------------------------------------------------------
RAW RESPONSE:
```sql
SELECT object_id, organelle_type, volume FROM organelles
WHERE organelle_type='mitochondria' ORDER BY volume DESC LIMIT 3;
```
--------------------------------------------------------------------------------
CLEANED SQL:
SELECT object_id, organelle_type, volume FROM organelles
WHERE organelle_type='mitochondria' ORDER BY volume DESC LIMIT 3;
================================================================================
```

### 3. Layer Mapping (if applicable)

For visualization queries, shows how organelle types map to Neuroglancer layers:

```
================================================================================
[AI] LAYER MAPPING
================================================================================
PROMPT:
Match the organelle type to the correct Neuroglancer layer.

Organelle type: mitochondria

Available layers: ['fibsem-uint8', 'cell', 'ld', 'lyso', 'mito_filled', 'nuc', 'perox', 'yolk']

Which layer should be used to display this organelle type?
...
--------------------------------------------------------------------------------
RESPONSE:
mito_filled
================================================================================
```

## Benefits

1. **Debugging** - See exactly what the AI receives and returns
2. **Prompt Engineering** - Test different phrasings and see how AI interprets them
3. **Understanding** - Learn how fuzzy matching and intent classification work
4. **Model Comparison** - Compare different models' responses to the same prompts

## Example Session

```bash
$ pixi run python debug_query.py --verbose "can you display only the longest mitos?"

Initializing database...
  Database path: ./organelle_data/organelles.db
  Total organelles: 8821
  Available types: cell, lysosome, mitochondria, nucleus, peroxisome, yolk

Initializing query agent with model: qwen2.5-coder:3b
  Verbose mode: ENABLED

================================================================================
QUERY: can you display only the longest mitos?
================================================================================

[1] Processing query...

[AI] INTENT CLASSIFICATION
... (full prompt and response shown) ...

[AI] SQL GENERATION
... (full prompt and response shown) ...

[AI] LAYER MAPPING
... (full prompt and response shown) ...

[Results and visualization command shown]
================================================================================
```

## Non-Verbose Mode

Without the flag, you only see the final results:

```bash
$ pixi run python debug_query.py "show me the 3 largest mitos"

[QUERY_AGENT] Classified intent: visualization
[QUERY_AGENT] Generated SQL: SELECT object_id, organelle_type, volume FROM ...
[QUERY_AGENT] AI mapped 'mitochondria' → 'mito_filled'

[Results shown]
```

## Programmatic Usage

You can also enable verbose mode when creating QueryAgent directly:

```python
from query_agent import QueryAgent

agent = QueryAgent(db, model="qwen2.5-coder:3b", verbose=True)
result = agent.process_query("show me the 3 largest mitos")
```

## Tips

1. **Prompt Engineering** - Use verbose mode to refine your prompts and see how the AI responds
2. **Model Testing** - Compare outputs from different models (qwen2.5-coder:3b vs nemotron, etc.)
3. **Debugging Failures** - When queries fail, verbose mode shows where things went wrong
4. **Learning** - Great way to understand how the AI-driven system works internally

## Performance Note

Verbose mode adds minimal overhead - just the time to print the text. The AI model calls are the same whether verbose is on or off.
