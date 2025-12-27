"""
Query Agent for Natural Language to SQL

Uses Ollama/Nemotron to convert user queries to SQL and process results.
"""

import re
import os
from typing import Dict, Any, List, Optional
import ollama
from organelle_db import OrganelleDatabase
from ollama_manager import ensure_ollama_running, preload_model


class QueryAgent:
    """AI-powered query agent for organelle database."""

    # Keywords that indicate navigation queries
    NAVIGATION_KEYWORDS = [
        'take me to',
        'go to',
        'show me location',
        'navigate to',
        'show location',
        'where is',
        'find location'
    ]

    # Dangerous SQL keywords to block
    BLOCKED_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER',
        'CREATE', 'TRUNCATE', 'REPLACE', 'GRANT', 'REVOKE'
    ]

    def __init__(self, db: OrganelleDatabase, model: str = "nemotron-3-nano", ng_tracker=None, verbose: bool = False):
        """
        Initialize query agent.

        Args:
            db: OrganelleDatabase instance
            model: Ollama model name (default: nemotron-3-nano)
            ng_tracker: Optional NG_StateTracker instance for layer discovery and state access
            verbose: Enable verbose logging of AI interactions
        """
        # Ensure Ollama is running before initializing
        if not ensure_ollama_running():
            print("[QUERY_AGENT] Warning: Failed to start Ollama, queries may fail", flush=True)
        else:
            # Preload the model to avoid cold-start issues
            preload_model(model)

        self.db = db
        self.model = model
        self.ng_tracker = ng_tracker
        self.verbose = verbose

        # Discover available layers at init
        self.available_layers = {}
        if ng_tracker:
            self.available_layers = ng_tracker.get_available_layers()
            print(f"[QUERY_AGENT] Discovered layers: {list(self.available_layers.keys())}", flush=True)

        print(f"[QUERY_AGENT] Initialized with model: {model}", flush=True)

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process natural language query with agent-driven multi-query detection.

        Args:
            user_query: Natural language query from user

        Returns:
            Dictionary with query results and metadata
        """
        import time
        start = time.time()

        if not user_query.strip():
            return {
                "type": "error",
                "answer": "Please provide a query."
            }

        try:
            # Ask agent whether to split query
            queries = self._detect_and_split_query(user_query)

            if len(queries) > 1:
                # Multi-query: process each separately
                sub_results = []
                for sq in queries:
                    result = self._process_single_query(sq)
                    sub_results.append(result)

                # Combine results
                combined = self._combine_results(queries, sub_results)
                combined['timing'] = {'total': time.time() - start}
                return combined
            else:
                # Single query
                result = self._process_single_query(queries[0])
                result['timing'] = {'total': time.time() - start}
                return result

        except Exception as e:
            print(f"[QUERY_AGENT] Error processing query: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {
                "type": "error",
                "answer": f"Error processing query: {str(e)}"
            }

    def _classify_query_type(self, query: str, ai_interactions: List[Dict[str, Any]] = None) -> str:
        """
        Classify query intent using AI.

        Args:
            query: User query string
            ai_interactions: Optional list to track AI interactions

        Returns:
            Query intent: 'navigation', 'visualization', or 'informational'
        """
        prompt = f"""Classify the intent of this user query about organelle data.

User Query: {query}

Intent Types:
1. navigation - User wants to go to/view a specific location (e.g., "take me to the biggest mito", "navigate to nucleus 5")
2. visualization - User wants to show/hide/filter specific objects in the viewer (e.g., "show only the 3 largest mitos", "display nuclei with volume > 1000", "highlight the smallest ER")
3. informational - User wants statistical information without changing the view (e.g., "how many mitos are there?", "what is the average volume?")

Respond with ONLY one word: navigation, visualization, or informational

Intent:"""

        try:
            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] INTENT CLASSIFICATION", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            intent = response["message"]["content"].strip().lower()

            if self.verbose:
                print("RESPONSE:", flush=True)
                print(intent, flush=True)
                print("="*80 + "\n", flush=True)

            # Track AI interaction
            if ai_interactions is not None:
                ai_interactions.append({
                    'type': 'intent_classification',
                    'prompt': prompt,
                    'response': intent,
                    'model': self.model
                })

            # Validate response
            if intent in ['navigation', 'visualization', 'informational']:
                print(f"[QUERY_AGENT] Classified intent: {intent}", flush=True)
                return intent
            else:
                print(f"[QUERY_AGENT] Invalid intent '{intent}', defaulting to informational", flush=True)
                return "informational"

        except Exception as e:
            print(f"[QUERY_AGENT] Error classifying query: {e}", flush=True)
            return "informational"

    def _generate_sql(self, user_query: str, previous_error: Optional[str] = None, ai_interactions: List[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate SQL query from natural language using Ollama.

        Args:
            user_query: Natural language query
            previous_error: Error message from previous attempt (for retry)

        Returns:
            SQL query string or None if generation failed
        """
        schema_desc = self.db.get_schema_description()
        available_types = self.db.get_available_organelle_types()

        # Build fuzzy matching note with available types
        fuzzy_matching_note = f"""
IMPORTANT: The user may use informal names for organelles.
Available organelle types in database: {', '.join(available_types)}

Examples of fuzzy name matching:
- "mitos" or "mito" → use 'mitochondria'
- "nuclei" or "nuc" → use 'nucleus'
- "ER" → use 'endoplasmic_reticulum'
- "lyso" → use 'lysosome'
- "perox" → use 'peroxisome'

Match the user's query to the closest available organelle type from the list above.
"""

        # Add interpretation guidance (let AI decide, not hardcoded rules)
        interpretation_guide = """
IMPORTANT - Understanding User Intent:

Common ambiguous terms and how to interpret them:
- "longest" - Consider what metric best represents length/extent for the organelle type
  * For mitochondria: 'lsp (nm)' measures morphological length along skeleton
  * For other organelles: 'volume' or bounding box dimensions may be most relevant

- "largest/biggest" - Usually refers to volume

- "thickest/thinnest" - For mitochondria, 'radius mean (nm)' measures thickness

- "branched/complex" - For mitochondria, 'num branches' indicates structural complexity

YOU MUST DECIDE which field best answers the user's question based on:
1. The organelle type mentioned
2. What metadata fields are available (see schema above)
3. What the user is likely asking for

If unsure, choose the most semantically appropriate field.
"""

        # Add error feedback if retrying
        error_feedback = ""
        if previous_error:
            error_feedback = f"""
PREVIOUS SQL ATTEMPT FAILED with error:
{previous_error}

Fix the SQL to address this error. Common issues:
- Incorrect json_extract syntax (use '$.field name with spaces')
- Missing parentheses in nested functions
- Wrong column names (check schema above)
- Incorrect SQLite syntax
"""

        prompt = f"""You are an expert at converting natural language to SQLite queries.

Database Schema:
{schema_desc}

{fuzzy_matching_note}

{interpretation_guide}

{error_feedback}

User Query: {user_query}

IMPORTANT RULES:
1. Generate ONLY a valid, complete SQLite SELECT query
2. The query MUST include FROM organelles
3. The query MUST end with a semicolon
4. No explanations, no markdown formatting, just the SQL
5. Query must be safe (no DROP, INSERT, UPDATE, DELETE)
6. Use proper SQLite syntax
7. Match user's organelle names to the available types listed above

Example Queries (20 examples covering various patterns):

# Basic queries
1. "What is the size of the biggest mito?"
   → SELECT object_id, organelle_type, volume FROM organelles WHERE organelle_type='mitochondria' ORDER BY volume DESC LIMIT 1;

2. "How many nuclei are there?"
   → SELECT COUNT(*) as count FROM organelles WHERE organelle_type='nucleus';

3. "Take me to the biggest nucleus"
   → SELECT object_id, organelle_type, volume, position_x, position_y, position_z FROM organelles WHERE organelle_type='nucleus' ORDER BY volume DESC LIMIT 1;

4. "What is the average volume of ER?"
   → SELECT AVG(volume) as average_volume FROM organelles WHERE organelle_type='endoplasmic_reticulum';

5. "Show only the 3 largest mitos"
   → SELECT object_id, organelle_type, volume FROM organelles WHERE organelle_type='mitochondria' ORDER BY volume DESC LIMIT 3;

# Metadata field queries (NEW - shows json_extract usage)
6. "What is the longest mitochondria?" (uses lsp, NOT volume)
   → SELECT object_id, organelle_type, json_extract(metadata, '$.lsp (nm)') as lsp FROM organelles WHERE organelle_type='mitochondria' ORDER BY json_extract(metadata, '$.lsp (nm)') DESC LIMIT 1;

7. "Find the thickest mito" (uses radius mean)
   → SELECT object_id, organelle_type, json_extract(metadata, '$.radius mean (nm)') as radius FROM organelles WHERE organelle_type='mitochondria' ORDER BY json_extract(metadata, '$.radius mean (nm)') DESC LIMIT 1;

8. "Show mitos with more than 5 branches"
   → SELECT object_id, organelle_type, json_extract(metadata, '$.num branches') as branches FROM organelles WHERE organelle_type='mitochondria' AND json_extract(metadata, '$.num branches') > 5;

9. "What is the average lsp of mitochondria?"
   → SELECT AVG(json_extract(metadata, '$.lsp (nm)')) as avg_lsp FROM organelles WHERE organelle_type='mitochondria';

10. "Display the 3 most branched mitos"
    → SELECT object_id, organelle_type, json_extract(metadata, '$.num branches') as branches FROM organelles WHERE organelle_type='mitochondria' ORDER BY json_extract(metadata, '$.num branches') DESC LIMIT 3;

# Range queries (NEW)
11. "Find mitos with volume greater than 100000"
    → SELECT object_id, organelle_type, volume FROM organelles WHERE organelle_type='mitochondria' AND volume > 100000;

12. "Show nuclei with volume between 1000 and 5000"
    → SELECT object_id, organelle_type, volume FROM organelles WHERE organelle_type='nucleus' AND volume BETWEEN 1000 AND 5000;

13. "Find lysosomes smaller than 500"
    → SELECT object_id, organelle_type, volume FROM organelles WHERE organelle_type='lysosome' AND volume < 500;

# Multiple organelle types (NEW)
14. "How many mitos and nuclei are there?"
    → SELECT organelle_type, COUNT(*) as count FROM organelles WHERE organelle_type IN ('mitochondria', 'nucleus') GROUP BY organelle_type;

15. "Show all peroxisomes and lysosomes"
    → SELECT object_id, organelle_type, volume FROM organelles WHERE organelle_type IN ('peroxisome', 'lysosome');

# Aggregations (NEW)
16. "What is the total volume of all mitos?"
    → SELECT SUM(volume) as total_volume FROM organelles WHERE organelle_type='mitochondria';

17. "Find the min and max volume of nuclei"
    → SELECT MIN(volume) as min_volume, MAX(volume) as max_volume FROM organelles WHERE organelle_type='nucleus';

# Navigation with metadata (NEW)
18. "Take me to the longest mito" (uses lsp for navigation)
    → SELECT object_id, organelle_type, json_extract(metadata, '$.lsp (nm)') as lsp, position_x, position_y, position_z FROM organelles WHERE organelle_type='mitochondria' ORDER BY json_extract(metadata, '$.lsp (nm)') DESC LIMIT 1;

# Specific objects (NEW)
19. "Show me nucleus 5"
    → SELECT object_id, organelle_type, volume, position_x, position_y, position_z FROM organelles WHERE organelle_type='nucleus' AND object_id='5';

20. "Describe mitochondria 1234"
    → SELECT object_id, organelle_type, volume, surface_area, position_x, position_y, position_z, metadata FROM organelles WHERE organelle_type='mitochondria' AND object_id='1234';

IMPORTANT:
- ALWAYS include BOTH object_id AND organelle_type in SELECT for any query that returns specific organelles
- For navigation queries (take me to, go to, etc.), MUST also include position_x, position_y, position_z
- For visualization queries (show, display, highlight), object_id and organelle_type are REQUIRED

UNITS AND COORDINATE SYSTEM:
- Volume is stored in nm³ (cubic nanometers) in the 'volume' COLUMN (not metadata!)
- Surface area is stored in nm² (square nanometers) in the 'surface_area' COLUMN (not metadata!)
- Position coordinates (position_x, position_y, position_z) are in nm (nanometers) as COLUMNS
- Metadata fields like 'lsp (nm)', 'radius mean (nm)' require json_extract(metadata, '$.field')
- Position coordinates are in (x, y, z) order matching the CSV files and Neuroglancer viewer

CRITICAL - Column vs Metadata:
- volume, surface_area, position_x/y/z, object_id, organelle_type are DIRECT COLUMNS - use them directly!
- Other fields like 'lsp (nm)', 'radius mean (nm)', 'num branches' are in metadata JSON - use json_extract()
- DO NOT extract volume or surface_area from metadata - they are direct columns!

Remember: Your response must be ONLY the SQL query, nothing else.

SQL Query:"""

        try:
            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] SQL GENERATION", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            sql_raw = response["message"]["content"].strip()

            if self.verbose:
                print("RAW RESPONSE:", flush=True)
                print(sql_raw, flush=True)
                print("-"*80, flush=True)

            # Clean up response (remove markdown, extra text)
            sql = self._clean_sql(sql_raw)

            if self.verbose:
                print("CLEANED SQL:", flush=True)
                print(sql, flush=True)
                print("="*80 + "\n", flush=True)

            # Track AI interaction
            if ai_interactions is not None:
                ai_interactions.append({
                    'type': 'sql_generation',
                    'prompt': prompt,
                    'response': sql_raw,
                    'cleaned_sql': sql,
                    'model': self.model,
                    'retry': previous_error is not None,
                    'previous_error': previous_error
                })

            print(f"[QUERY_AGENT] Generated SQL: {sql}", flush=True)
            return sql

        except Exception as e:
            print(f"[QUERY_AGENT] Error generating SQL: {e}", flush=True)
            return None

    def _clean_sql(self, sql: str) -> str:
        """
        Clean SQL response from LLM.

        Args:
            sql: Raw SQL from LLM

        Returns:
            Cleaned SQL query
        """
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)

        # Remove extra whitespace
        sql = sql.strip()

        # If the AI returned multiple SQL queries (separated by semicolons), take the first one
        # This can happen when the AI formats queries nicely across multiple lines
        if ';' in sql:
            # Split by semicolon and take the first complete query
            queries = sql.split(';')
            for query in queries:
                query = query.strip()
                if query.upper().startswith('SELECT') and 'FROM' in query.upper():
                    # Found a complete query
                    sql = query + ';'
                    break

        # Collapse multi-line formatted SQL into single line
        # This handles cases where AI formats SQL nicely with line breaks
        sql = ' '.join(sql.split())

        return sql

    def _validate_sql(self, sql: str) -> bool:
        """
        Validate SQL query is safe.

        Args:
            sql: SQL query string

        Returns:
            True if safe, False otherwise
        """
        sql_upper = sql.upper()

        # Check for blocked keywords
        for keyword in self.BLOCKED_KEYWORDS:
            if keyword in sql_upper:
                print(f"[QUERY_AGENT] Blocked dangerous keyword: {keyword}", flush=True)
                return False

        # Must be a SELECT query
        if not sql_upper.strip().startswith('SELECT'):
            print(f"[QUERY_AGENT] Query must start with SELECT", flush=True)
            return False

        # Must have FROM clause
        if 'FROM' not in sql_upper:
            print(f"[QUERY_AGENT] Query must include FROM clause", flush=True)
            return False

        # Should reference the organelles table
        if 'ORGANELLES' not in sql_upper:
            print(f"[QUERY_AGENT] Query must reference 'organelles' table", flush=True)
            return False

        # Block SQL comments (potential injection)
        if '--' in sql or '/*' in sql:
            print(f"[QUERY_AGENT] Blocked SQL comments", flush=True)
            return False

        return True

    def _validate_sql_syntax(self, sql: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL syntax using SQLite EXPLAIN.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Use EXPLAIN to validate without executing
            explain_sql = f"EXPLAIN {sql}"
            self.db.execute_query(explain_sql)
            return (True, None)
        except Exception as e:
            error_msg = str(e)
            print(f"[QUERY_AGENT] SQL syntax validation failed: {error_msg}", flush=True)
            return (False, error_msg)

    def _generate_sql_with_retry(self, user_query: str, max_retries: int = 2, ai_interactions: List[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate SQL with retry mechanism.

        Args:
            user_query: Natural language query
            max_retries: Maximum number of retry attempts

        Returns:
            Valid SQL query string or None
        """
        last_error = None

        for attempt in range(max_retries + 1):
            # Generate SQL with error feedback
            sql = self._generate_sql(user_query, previous_error=last_error if attempt > 0 else None, ai_interactions=ai_interactions)

            if not sql:
                return None

            # Validate safety
            if not self._validate_sql(sql):
                print(f"[QUERY_AGENT] Unsafe SQL on attempt {attempt + 1}", flush=True)
                continue

            # Validate syntax
            is_valid, error_msg = self._validate_sql_syntax(sql)
            if not is_valid:
                print(f"[QUERY_AGENT] Invalid SQL attempt {attempt + 1}/{max_retries + 1}", flush=True)
                last_error = error_msg
                if attempt < max_retries:
                    print(f"[QUERY_AGENT] Retrying with error feedback...", flush=True)
                    continue
                else:
                    return None

            # Try executing
            try:
                self.db.execute_query(sql)
                return sql
            except Exception as e:
                error_msg = str(e)
                print(f"[QUERY_AGENT] Execution failed attempt {attempt + 1}/{max_retries + 1}: {error_msg}", flush=True)
                last_error = error_msg
                if attempt < max_retries:
                    print(f"[QUERY_AGENT] Retrying...", flush=True)
                    continue
                else:
                    return None

        return None

    def _process_single_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a single query (main query processing logic).

        Args:
            user_query: Natural language query

        Returns:
            Query result dictionary
        """
        import time
        timing = {}

        # Track AI interactions for verbose logging
        ai_interactions = []

        try:
            # Classify query type
            query_type = self._classify_query_type(user_query, ai_interactions)

            # Generate SQL with retry mechanism and timing
            start_sql = time.time()
            sql = self._generate_sql_with_retry(user_query, max_retries=2, ai_interactions=ai_interactions)
            timing['sql_generation'] = time.time() - start_sql

            if not sql:
                return {
                    "type": "error",
                    "answer": "Could not generate a valid query. Try asking something like: 'What is the size of the biggest mito?'"
                }

            # Execute query with timing (validation already done in retry wrapper)
            start_exec = time.time()
            results = self.db.execute_query(sql)
            timing['query_execution'] = time.time() - start_exec

            # Handle empty results
            if not results:
                available_types = self.db.get_available_organelle_types()
                return {
                    "type": query_type,
                    "sql": sql,
                    "results": [],
                    "answer": f"No results found. Available organelle types: {', '.join(available_types)}",
                    "timing": timing
                }

            # Handle based on query type
            result = None
            if query_type == "informational":
                result = self._handle_informational_query(user_query, sql, results, ai_interactions)
            elif query_type == "navigation":
                result = self._handle_navigation_query(user_query, sql, results, ai_interactions)
            elif query_type == "visualization":
                result = self._handle_visualization_query(user_query, sql, results, ai_interactions)
            else:
                # Default to informational
                result = self._handle_informational_query(user_query, sql, results, ai_interactions)

            # Always add AI interactions to result for verbose logging in UI
            if ai_interactions:
                result['ai_interactions'] = ai_interactions
                result['model'] = self.model

            return result

        except Exception as e:
            print(f"[QUERY_AGENT] Error processing single query: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {
                "type": "error",
                "answer": f"Error processing query: {str(e)}"
            }

    def _combine_results(self, queries: List[str], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from multiple sub-queries.

        Args:
            queries: List of query strings
            results: List of result dictionaries

        Returns:
            Combined result dictionary
        """
        answers = []
        all_results = []
        sql_queries = []
        visualizations = []
        navigations = []

        for i, (query, result) in enumerate(zip(queries, results), 1):
            # Collect answers
            answers.append(f"{i}. {result.get('answer', 'No answer')}")

            # Collect results
            all_results.extend(result.get('results', []))

            # Collect SQL queries
            if result.get('sql'):
                sql_queries.append(f"-- Query {i}: {query}\n{result['sql']}")

            # Collect visualizations
            if result.get('visualization'):
                visualizations.append(result['visualization'])

            # Collect navigations
            if result.get('navigation'):
                navigations.append(result['navigation'])

        # Determine combined type
        types = [r.get('type') for r in results]
        combined_type = 'visualization' if 'visualization' in types else 'informational'

        # Build combined result
        combined = {
            'type': combined_type,
            'results': all_results,
            'sub_queries': queries,
            'sub_results': results
        }

        # Add SQL queries if any
        if sql_queries:
            combined['sql'] = '\n\n'.join(sql_queries)

        # Use AI to generate combined response based on type
        if combined_type == 'visualization' and visualizations:
            # Use AI to intelligently combine visualization commands
            viz_result = self._combine_visualization_with_ai(queries, results, visualizations, all_results)
            combined['answer'] = viz_result.get('answer', '\n'.join(answers))
            combined['visualization'] = viz_result.get('visualization')
        elif navigations:
            # Navigation: use first one
            combined['navigation'] = navigations[0]
            combined['answer'] = '\n'.join(answers)
        else:
            # Informational: use AI to combine answers
            combined['answer'] = self._combine_answers_with_ai(queries, results) if len(results) > 1 else answers[0] if answers else "No answer"

        return combined

    def _combine_visualization_with_ai(
        self,
        queries: List[str],
        results: List[Dict[str, Any]],
        visualizations: List[Dict[str, Any]],
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use AI to intelligently combine multiple visualization commands.

        Args:
            queries: List of sub-query strings
            results: List of result dictionaries
            visualizations: List of visualization commands from sub-queries
            all_results: Combined database results

        Returns:
            Dictionary with 'answer' and 'visualization' fields
        """
        import json

        # If only one visualization, return it as-is
        if len(visualizations) == 1:
            answer = next((r.get('answer') for r in results if r.get('visualization')), "Visualization generated")
            return {
                'answer': answer,
                'visualization': visualizations[0]
            }

        # Multiple visualizations - use AI to combine
        viz_info = []
        for i, result in enumerate(results):
            if 'visualization' in result:
                viz_info.append({
                    'query': queries[i],
                    'visualization': result['visualization'],
                    'answer': result.get('answer', ''),
                    'num_objects': len(result.get('results', []))
                })

        prompt = f"""You are combining multiple visualization commands for Neuroglancer.

Sub-queries and their visualizations:
{json.dumps(viz_info, indent=2)}

IMPORTANT: Check if all visualizations use the SAME layer name:
- If YES (all same layer): Combine all segment IDs into a single layer command
- If NO (different layers): Keep as separate layer commands (return as a list)

Task:
1. Check if all layer_name values are the same
2. If same layer: combine all segment IDs into one command
3. If different layers: create separate commands for each layer
4. Create a natural language answer describing what will be shown (mention all organelles)

Respond in JSON format:
- For SAME layer:
{{
    "layer_commands": [
        {{
            "layer_name": "layer_name",
            "segment_ids": ["id1", "id2", ...],
            "action": "show_only"
        }}
    ],
    "answer": "Natural language answer"
}}

- For DIFFERENT layers:
{{
    "layer_commands": [
        {{
            "layer_name": "layer1",
            "segment_ids": ["id1"],
            "action": "show_only"
        }},
        {{
            "layer_name": "layer2",
            "segment_ids": ["id2"],
            "action": "show_only"
        }}
    ],
    "answer": "Natural language answer mentioning both/all"
}}

JSON:"""

        try:
            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] VISUALIZATION COMBINATION", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response["message"]["content"].strip()

            if self.verbose:
                print("RESPONSE:", flush=True)
                print(response_text, flush=True)
                print("="*80 + "\n", flush=True)

            # Clean and parse JSON
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            viz_combined = json.loads(response_text)

            layer_commands = viz_combined.get('layer_commands', [])

            print(f"[QUERY_AGENT] AI combined {len(visualizations)} visualizations into {len(layer_commands)} layer command(s)", flush=True)

            # Return in format expected by the system
            # If multiple layers, we return visualization as a list of commands
            # If single layer, return as a single command for backward compatibility
            if len(layer_commands) == 1:
                viz_result = {
                    'answer': viz_combined.get('answer', 'Visualization generated'),
                    'visualization': layer_commands[0]
                }
            else:
                viz_result = {
                    'answer': viz_combined.get('answer', 'Visualization generated'),
                    'visualization': layer_commands  # List of commands for multi-layer
                }

            return viz_result

        except Exception as e:
            print(f"[QUERY_AGENT] Error combining visualizations with AI: {e}, using fallback", flush=True)
            import traceback
            traceback.print_exc()

            # Fallback: manually combine
            # Group by layer name
            layer_groups = {}
            for viz in visualizations:
                layer_name = viz.get('layer_name', 'unknown')
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = []
                layer_groups[layer_name].extend(viz.get('segment_ids', []))

            # Create layer commands
            layer_commands = []
            for layer_name, segment_ids in layer_groups.items():
                layer_commands.append({
                    'layer_name': layer_name,
                    'segment_ids': segment_ids,
                    'action': 'show_only'
                })

            answers = [f"{i+1}. {r.get('answer', 'No answer')}" for i, r in enumerate(results) if r.get('type') == 'visualization']

            # Return single command or list depending on number of layers
            if len(layer_commands) == 1:
                return {
                    'answer': '\n'.join(answers),
                    'visualization': layer_commands[0]
                }
            else:
                return {
                    'answer': '\n'.join(answers),
                    'visualization': layer_commands
                }

    def _combine_answers_with_ai(self, queries: List[str], results: List[Dict[str, Any]]) -> str:
        """
        Use AI to create a combined natural language answer from multiple sub-query results.

        Args:
            queries: List of sub-query strings
            results: List of result dictionaries

        Returns:
            Combined natural language answer
        """
        import json

        # Build context
        query_info = []
        for i, (query, result) in enumerate(zip(queries, results), 1):
            query_info.append({
                'query': query,
                'answer': result.get('answer', 'No answer'),
                'sample_data': result.get('results', [])[:2]  # Limit to avoid token overflow
            })

        prompt = f"""You are combining answers from multiple related queries about organelle data.

Queries and individual answers:
{json.dumps(query_info, indent=2)}

Task:
Combine these answers into a single, cohesive natural language response.
- Be concise (2-3 sentences max)
- Include all key information from each query
- Use natural language, not a numbered list

Combined answer:"""

        try:
            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] ANSWER COMBINATION", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["message"]["content"].strip()

            if self.verbose:
                print("RESPONSE:", flush=True)
                print(answer, flush=True)
                print("="*80 + "\n", flush=True)

            print(f"[QUERY_AGENT] AI combined {len(results)} answers", flush=True)
            return answer

        except Exception as e:
            print(f"[QUERY_AGENT] Error combining answers with AI: {e}, using fallback", flush=True)
            # Fallback: numbered list
            return '\n'.join([f"{i+1}. {r.get('answer', 'No answer')}" for i, r in enumerate(results)])

    def _detect_and_split_query(self, user_query: str) -> List[str]:
        """
        Ask AI agent whether query should be split into multiple queries.
        Returns list of queries (single item if no split needed).

        Args:
            user_query: User's natural language query

        Returns:
            List of query strings
        """
        prompt = f"""Analyze this user query about organelle data.

User Query: {user_query}

Does this query contain MULTIPLE DISTINCT QUESTIONS that should be answered separately?

Examples of queries that SHOULD be split:
- "show largest mito and smallest nucleus" (2 separate questions)
- "what is biggest cell and how many lysosomes" (2 separate questions)
- "display 3 biggest mitos and 2 smallest nuclei" (2 separate questions)

Examples of queries that should NOT be split:
- "show largest mito" (single question)
- "how many mitos and nuclei are there" (single question asking for counts of multiple types)
- "show mitos and nuclei with volume > 1000" (single question with one condition)

If the query should be split, respond with:
SPLIT
1. [first query]
2. [second query]
...

If the query should NOT be split, respond with:
KEEP
[original query]

Your response:"""

        try:
            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] QUERY SPLIT DETECTION", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response["message"]["content"].strip()

            if self.verbose:
                print("RESPONSE:", flush=True)
                print(response_text, flush=True)
                print("="*80 + "\n", flush=True)

            # Parse response
            if response_text.startswith("SPLIT"):
                # Extract queries
                queries = []
                import re
                for line in response_text.split('\n')[1:]:  # Skip "SPLIT" line
                    match = re.match(r'^[\d\-\*\.]+\s+(.+)$', line.strip())
                    if match:
                        queries.append(match.group(1).strip())

                if queries:
                    print(f"[QUERY_AGENT] Agent split into {len(queries)} queries: {queries}", flush=True)
                    return queries
                else:
                    print(f"[QUERY_AGENT] Agent said split but couldn't parse, using original", flush=True)
                    return [user_query]
            else:
                # Keep as single query
                print(f"[QUERY_AGENT] Agent determined query should not be split", flush=True)
                return [user_query]

        except Exception as e:
            print(f"[QUERY_AGENT] Split detection failed: {e}, using original", flush=True)
            return [user_query]

    def _handle_informational_query(
        self,
        user_query: str,
        sql: str,
        results: List[Dict[str, Any]],
        ai_interactions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle informational query (no navigation).

        Args:
            user_query: Original user query
            sql: Generated SQL
            results: Query results
            ai_interactions: Optional list to track AI interactions

        Returns:
            Response dictionary
        """
        # Format answer
        answer = self._format_answer(results, user_query, ai_interactions)

        return {
            "type": "informational",
            "sql": sql,
            "results": results,
            "answer": answer
        }

    def _get_ng_state(self) -> Dict[str, Any]:
        """
        Get current Neuroglancer state.

        Returns:
            Neuroglancer state as dictionary
        """
        if not self.ng_tracker:
            return {}

        try:
            with self.ng_tracker.viewer.txn() as s:
                return s.to_json()
        except Exception as e:
            print(f"[QUERY_AGENT] Error getting NG state: {e}", flush=True)
            return {}

    def _extract_voxel_size_from_state(self, ng_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract voxel size and axis order from Neuroglancer state dimensions.

        Args:
            ng_state: Neuroglancer state dictionary

        Returns:
            Dictionary with:
                - 'axis_order': List of axis names in order (e.g., ['x', 'y', 'z'] or ['z', 'y', 'x'])
                - 'voxel_sizes': Dictionary mapping axis name to voxel size in nm
        """
        # Default for C. elegans dataset (x, y, z order)
        default_result = {
            'axis_order': ['x', 'y', 'z'],
            'voxel_sizes': {'x': 8.0, 'y': 8.0, 'z': 8.0}
        }

        if not ng_state or 'dimensions' not in ng_state:
            print(f"[QUERY_AGENT] No dimensions in state, using default", flush=True)
            return default_result

        dimensions = ng_state['dimensions']

        try:
            # Neuroglancer dimensions can be ordered arbitrarily
            # Extract axis order from the state (order matters!)
            if isinstance(dimensions, dict):
                axis_order = list(dimensions.keys())
            else:
                print(f"[QUERY_AGENT] Dimensions not a dict, using default", flush=True)
                return default_result

            voxel_sizes = {}
            for axis in axis_order:
                dim_info = dimensions[axis]
                if isinstance(dim_info, list) and len(dim_info) >= 2:
                    # Format: [scale, unit]
                    scale = float(dim_info[0])
                    unit = dim_info[1]

                    # Convert to nanometers
                    if unit == 'm':
                        voxel_size_nm = scale * 1e9
                    elif unit == 'nm':
                        voxel_size_nm = scale
                    else:
                        print(f"[QUERY_AGENT] Unknown unit '{unit}' for axis {axis}, assuming meters", flush=True)
                        voxel_size_nm = scale * 1e9

                    voxel_sizes[axis] = voxel_size_nm
                else:
                    # Fallback to default
                    voxel_sizes[axis] = 8.0

            result = {
                'axis_order': axis_order,
                'voxel_sizes': voxel_sizes
            }

            print(f"[QUERY_AGENT] Extracted from state: axis_order={axis_order}, voxel_sizes={voxel_sizes}", flush=True)
            return result

        except Exception as e:
            print(f"[QUERY_AGENT] Error extracting voxel info: {e}, using default", flush=True)
            import traceback
            traceback.print_exc()
            return default_result

    def _generate_navigation_command_with_ai(
        self,
        result: Dict[str, Any],
        ng_state: Dict[str, Any],
        user_query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use AI to generate navigation command by converting coordinates and determining zoom.

        Args:
            result: SQL result with position_x/y/z in nanometers and volume
            ng_state: Full Neuroglancer state (contains dimensions with voxel sizes)
            user_query: Original user query

        Returns:
            Dictionary with 'position' (voxel coords), 'scale', and 'answer', or None if failed
        """
        import json

        # Extract position from result (in nanometers)
        position_x = result.get('position_x')
        position_y = result.get('position_y')
        position_z = result.get('position_z')

        if position_x is None or position_y is None or position_z is None:
            return None

        # Extract relevant state info (dimensions with voxel sizes)
        dimensions = ng_state.get('dimensions', {})

        # Simplified prompt - be very direct and concrete
        voxel_x = dimensions.get('x', [8e-9, 'm'])[0] * 1e9 if dimensions.get('x') else 8.0
        voxel_y = dimensions.get('y', [8e-9, 'm'])[0] * 1e9 if dimensions.get('y') else 8.0
        voxel_z = dimensions.get('z', [8e-9, 'm'])[0] * 1e9 if dimensions.get('z') else 8.0

        prompt = f"""Convert coordinates and generate navigation command.

Position in nanometers: ({position_x:.1f}, {position_y:.1f}, {position_z:.1f}) nm
Voxel size: ({voxel_x:.1f}, {voxel_y:.1f}, {voxel_z:.1f}) nm/voxel
Volume: {result.get('volume', 'N/A')} nm³
Object: {result.get('organelle_type', 'organelle')} {result.get('object_id', 'unknown')}

Calculate:
1. Voxel coordinates = position_nm / voxel_size
2. Scale based on volume: <1000→10, <10000→30, <100000→100, <500000→200, ≥500000→300

Return ONLY this JSON (no explanation):
{{"position": [x_voxel, y_voxel, z_voxel], "scale": <int>, "answer": "Taking you to [type] [id] at ([x], [y], [z]) nm with volume [vol] nm³"}}"""

        try:
            print(f"[QUERY_AGENT] Calling AI to generate navigation command (model={self.model})...", flush=True)

            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] NAVIGATION COMMAND GENERATION", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            import time
            start = time.time()

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": 500,  # Limit response length
                    "temperature": 0.1,  # More deterministic
                }
            )

            elapsed = time.time() - start
            print(f"[QUERY_AGENT] AI navigation call completed in {elapsed:.2f}s", flush=True)

            response_text = response["message"]["content"].strip()

            print(f"[QUERY_AGENT] AI response text: '{response_text[:200]}'...", flush=True)

            if self.verbose:
                print("RESPONSE:", flush=True)
                print(response_text, flush=True)
                print("="*80 + "\n", flush=True)

            # Check if response is empty
            if not response_text:
                print(f"[QUERY_AGENT] AI returned empty response!", flush=True)
                return None

            # Clean and parse JSON
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            nav_command = json.loads(response_text)

            # Validate response structure
            if 'position' not in nav_command or 'scale' not in nav_command:
                print(f"[QUERY_AGENT] AI navigation response missing required fields", flush=True)
                return None

            print(f"[QUERY_AGENT] AI generated navigation: position={nav_command['position']}, scale={nav_command['scale']}", flush=True)
            return nav_command

        except Exception as e:
            print(f"[QUERY_AGENT] Error generating navigation with AI: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    def _handle_navigation_query(
        self,
        user_query: str,
        sql: str,
        results: List[Dict[str, Any]],
        ai_interactions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle navigation query (update viewer position).
        Uses AI to compute navigation commands based on NG state.

        Args:
            user_query: Original user query
            sql: Generated SQL
            results: Query results

        Returns:
            Response dictionary with navigation commands
        """
        # Get first result
        result = results[0]

        # Extract position
        position_x = result.get('position_x')
        position_y = result.get('position_y')
        position_z = result.get('position_z')

        if position_x is None or position_y is None or position_z is None:
            return {
                "type": "error",
                "answer": "Position data not available for this organelle.",
                "sql": sql,
                "results": results
            }

        # Get Neuroglancer state to extract voxel size and axis order
        ng_state = self._get_ng_state() if self.ng_tracker else {}

        # Extract voxel size and axis order from state
        voxel_info = self._extract_voxel_size_from_state(ng_state)
        axis_order = voxel_info['axis_order']
        voxel_sizes = voxel_info['voxel_sizes']

        # Database stores positions as (position_x, position_y, position_z) = (x, y, z)
        # We need to convert these to voxels in the order expected by Neuroglancer state
        position_nm = {
            'x': position_x,
            'y': position_y,
            'z': position_z
        }

        # Convert to voxel coordinates in the order specified by the state
        voxel_position = []
        for axis in axis_order:
            if axis in position_nm and axis in voxel_sizes:
                voxel_coord = int(position_nm[axis] / voxel_sizes[axis])
                voxel_position.append(voxel_coord)
            else:
                print(f"[QUERY_AGENT] Warning: axis '{axis}' not found in position or voxel_sizes", flush=True)
                voxel_position.append(0)

        # Create natural language answer with units
        organelle_type = result.get('organelle_type', 'organelle').replace('_', ' ')
        object_id = result.get('object_id', 'unknown')
        volume = result.get('volume', 0)

        if volume and volume > 0:
            vol_str = f"{volume:.2e}" if volume >= 1e6 else f"{volume:.1f}"
            answer = f"Taking you to {organelle_type} {object_id} at position ({position_x:.1f}, {position_y:.1f}, {position_z:.1f}) nm with volume {vol_str} nm³"
        else:
            answer = f"Taking you to {organelle_type} {object_id} at position ({position_x:.1f}, {position_y:.1f}, {position_z:.1f}) nm"

        nav_command = {
            "position": voxel_position,
            "answer": answer
        }

        print(f"[QUERY_AGENT] Generated navigation: nm=(x:{position_x:.1f}, y:{position_y:.1f}, z:{position_z:.1f}) → voxel={voxel_position} (axis_order={axis_order})", flush=True)

        # Add metadata
        nav_command["object_id"] = result.get('object_id', 'unknown')

        return {
            "type": "navigation",
            "sql": sql,
            "results": results,
            "answer": nav_command.get("answer", "Navigating..."),
            "navigation": {
                "position": nav_command["position"],
                "object_id": nav_command["object_id"]
            }
        }

    def _handle_visualization_query(
        self,
        user_query: str,
        sql: str,
        results: List[Dict[str, Any]],
        ai_interactions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle visualization query using AI to determine state updates.

        Args:
            user_query: Original user query
            sql: Generated SQL
            results: Query results
            ai_interactions: Optional list to track AI interactions

        Returns:
            Response dictionary with visualization commands
        """
        if not results:
            return {
                "type": "error",
                "answer": "No objects found to visualize.",
                "sql": sql,
                "results": results
            }

        # Use AI to generate visualization state updates
        viz_updates = self._generate_visualization_state_update(
            user_query=user_query,
            sql_results=results,
            ai_interactions=ai_interactions
        )

        if viz_updates.get("error"):
            return {
                "type": "error",
                "answer": viz_updates["error"],
                "sql": sql,
                "results": results
            }

        return {
            "type": "visualization",
            "sql": sql,
            "results": results,
            "answer": viz_updates.get("answer", "Updating visualization..."),
            "visualization": viz_updates.get("commands", [])
        }

    def _generate_visualization_state_update(
        self,
        user_query: str,
        sql_results: List[Dict[str, Any]],
        ai_interactions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use AI to generate visualization state updates based on query and results.

        This agent-driven approach allows flexible interpretation of:
        - Which layers to show/hide
        - Whether to replace or add to current selection
        - How to format segment IDs
        - What action to take (show_only, add, remove)

        Args:
            user_query: Original user query
            sql_results: Results from SQL query
            ai_interactions: Optional list to track AI interactions

        Returns:
            Dictionary with 'commands' (list of viz commands) and 'answer' (natural language)
        """
        import json

        # Get current NG state if available
        current_state = {}
        if self.ng_tracker:
            try:
                with self.ng_tracker.viewer.txn() as s:
                    state_json = s.to_json()
                    # Extract relevant parts: visible layers and selected segments
                    current_state = {
                        "layers": []
                    }
                    if "layers" in state_json:
                        for layer in state_json["layers"]:
                            layer_info = {
                                "name": layer.get("name"),
                                "type": layer.get("type"),
                                "visible": layer.get("visible", True)
                            }
                            if "segments" in layer:
                                layer_info["segments"] = layer["segments"]
                            current_state["layers"].append(layer_info)
            except Exception as e:
                print(f"[QUERY_AGENT] Could not get current NG state: {e}", flush=True)

        # Build available layers info
        available_layers = list(self.available_layers.keys()) if self.available_layers else []

        # Limit results to avoid token overflow
        max_results = 10
        results_to_show = sql_results[:max_results]

        prompt = f"""You are helping visualize organelle data in Neuroglancer based on a user query and SQL results.

User Query: {user_query}

SQL Results (showing {len(results_to_show)} of {len(sql_results)} objects):
{json.dumps(results_to_show, indent=2)}

Available Neuroglancer Layers: {available_layers}

Current Neuroglancer State:
{json.dumps(current_state, indent=2)}

Instructions:
1. Based on the user query and SQL results, determine what visualization updates are needed
2. Generate visualization commands that update the Neuroglancer state appropriately
3. Consider the semantics of the query:
   - "show X" / "show me X" → show_only (hide other segments in same layer)
   - "also show X" / "add X" → add (keep existing segments visible)
   - "hide X" / "remove X" → remove
   - "show X and Y" → show_only with both X and Y
4. Map organelle types to the correct layer names (be flexible with matching)
5. Extract segment IDs from the SQL results' object_id field
   - Segment IDs should be stringified object IDs (e.g., if object_id=123, use "123")
6. Create a natural language answer describing what will be shown (1-2 sentences max)

Examples of layer matching:
- mitochondria → mito
- nucleus → nuc
- endoplasmic_reticulum → er
- lipid_droplet → ld
- lysosome → lyso
- peroxisome → perox
- yolk → yolk

Output Format (respond ONLY with valid JSON, no other text):
{{
    "commands": [
        {{
            "layer_name": "mito",
            "segment_ids": ["123", "456"],
            "action": "show_only"
        }}
    ],
    "answer": "Showing the 2 largest mitochondria with volumes 3.2e11 nm³ and 2.8e11 nm³"
}}

Response:"""

        try:
            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] VISUALIZATION STATE UPDATE", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json"  # Request JSON output
            )

            response_text = response["message"]["content"].strip()

            if self.verbose:
                print("RESPONSE:", flush=True)
                print(response_text, flush=True)
                print("="*80 + "\n", flush=True)

            # Track AI interaction
            if ai_interactions is not None:
                ai_interactions.append({
                    'type': 'visualization_state_update',
                    'prompt': prompt,
                    'response': response_text,
                    'model': self.model
                })

            # Parse JSON response
            try:
                viz_data = json.loads(response_text)
                commands = viz_data.get("commands", [])
                answer = viz_data.get("answer", "Updating visualization...")

                print(f"[QUERY_AGENT] AI generated {len(commands)} visualization commands", flush=True)
                return {
                    "commands": commands,
                    "answer": answer
                }

            except json.JSONDecodeError as e:
                print(f"[QUERY_AGENT] Failed to parse AI JSON response: {e}", flush=True)
                print(f"[QUERY_AGENT] Response was: {response_text}", flush=True)
                # Fall back to hardcoded approach
                return self._fallback_visualization_update(sql_results)

        except Exception as e:
            print(f"[QUERY_AGENT] Error in AI visualization update: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._fallback_visualization_update(sql_results)

    def _fallback_visualization_update(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback visualization update when AI fails.

        Args:
            results: SQL query results

        Returns:
            Visualization commands dict
        """
        # Extract segment IDs and organelle type from results
        segment_ids = []
        organelle_type = None

        for result in results:
            obj_id = result.get('object_id')
            org_type = result.get('organelle_type')

            if not organelle_type and org_type:
                organelle_type = org_type

            if obj_id is not None:
                # Convert to int first to handle floats like 2.0 -> "2" instead of "2.0"
                try:
                    obj_id_str = str(int(float(obj_id)))
                except (ValueError, TypeError):
                    obj_id_str = str(obj_id)

                # Segment IDs are just stringified object IDs
                segment_ids.append(obj_id_str)

        if not segment_ids:
            return {
                "error": "No objects found to visualize."
            }

        # Map organelle type to layer name using fallback
        layer_name = self._get_layer_name_fallback(organelle_type)

        # Create simple answer
        count = len(segment_ids)
        type_name = organelle_type.replace('_', ' ') if organelle_type else "organelle"
        if count == 1:
            vol = results[0].get('volume')
            vol_str = f" with volume {vol:.2e} nm³" if vol else ""
            answer = f"Showing {type_name} {segment_ids[0]}{vol_str}"
        else:
            answer = f"Showing {count} {type_name} objects"

        return {
            "commands": [{
                "layer_name": layer_name,
                "segment_ids": segment_ids,
                "action": "show_only"
            }],
            "answer": answer
        }

    def _get_layer_name(self, organelle_type: str) -> str:
        """
        Map organelle type to Neuroglancer layer name using AI.

        Falls back to hardcoded mappings if AI fails or no layers discovered.

        Args:
            organelle_type: Organelle type from database

        Returns:
            Layer name for Neuroglancer
        """
        if not self.available_layers:
            # No Neuroglancer available, use fallback
            return self._get_layer_name_fallback(organelle_type)

        prompt = f"""Match the organelle type to the correct Neuroglancer layer.

Organelle type: {organelle_type}

Available layers: {list(self.available_layers.keys())}

Which layer should be used to display this organelle type?
Be flexible with matching - for example:
- "mitochondria" → "mito_filled" or "mito_seg"
- "nucleus" → "nuc" or "nucleus_seg"
- "endoplasmic_reticulum" → "er_seg" or "er"
- "yolk_filled" → "yolk"

Respond with ONLY the layer name, nothing else.

Layer name:"""

        try:
            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] LAYER MAPPING", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
            layer_name = response["message"]["content"].strip()

            if self.verbose:
                print("RESPONSE:", flush=True)
                print(layer_name, flush=True)
                print("="*80 + "\n", flush=True)

            if layer_name in self.available_layers:
                print(f"[QUERY_AGENT] AI mapped '{organelle_type}' → '{layer_name}'", flush=True)
                return layer_name
            else:
                print(f"[QUERY_AGENT] AI suggested invalid layer '{layer_name}', using fallback", flush=True)
                return self._get_layer_name_fallback(organelle_type)

        except Exception as e:
            print(f"[QUERY_AGENT] Error in AI layer mapping: {e}, using fallback", flush=True)
            return self._get_layer_name_fallback(organelle_type)

    def _get_layer_name_fallback(self, organelle_type: str) -> str:
        """
        Hardcoded fallback mapping for layer names.

        Args:
            organelle_type: Organelle type from database

        Returns:
            Layer name for Neuroglancer
        """
        # Mapping from database organelle types to layer names
        # For C. elegans dataset:
        layer_mapping = {
            'mitochondria': 'mito_filled',
            'nucleus': 'nuc',
            'lysosome': 'lyso',
            'peroxisome': 'perox',
            'cell': 'cell',
            'yolk_filled': 'yolk',
            'lipid_droplet': 'ld',
            # For HeLa dataset:
            'endoplasmic_reticulum': 'er_seg',
            'golgi_apparatus': 'golgi_seg',
            'vesicle': 'vesicle_seg',
            'endosome': 'endo_seg',
        }

        return layer_mapping.get(organelle_type, organelle_type)

    def _format_visualization_answer(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        organelle_type: str,
        segment_ids: List[str],
        ai_interactions: List[Dict[str, Any]] = None
    ) -> str:
        """
        Format a natural language answer for visualization query using AI.

        Args:
            user_query: Original user query
            results: Query results
            organelle_type: Type of organelle (can be None)
            segment_ids: List of segment IDs to show

        Returns:
            Natural language answer
        """
        import json

        # Limit results to avoid token overflow
        max_results = 5
        results_to_show = results[:max_results]

        prompt = f"""You are responding to a user's visualization query about organelle data.

User Query: {user_query}

Action: Showing {len(segment_ids)} {organelle_type.replace('_', ' ') if organelle_type else 'organelle'} object(s) in the viewer

Query Results (showing up to {max_results} of {len(results)}):
{json.dumps(results_to_show, indent=2)}

Instructions:
1. Create a natural language response describing what will be shown in the viewer
2. Keep it to 1-2 sentences maximum
3. ALWAYS include units where relevant:
   - Volume: nm³
   - Surface area: nm²
   - Use scientific notation for large values (e.g., 3.05e11 nm³)
4. For single objects, mention key details (ID, volume, etc.)
5. For multiple objects, summarize (e.g., "3 largest mitochondria", "5 nuclei")
6. Use natural language, not technical jargon

Examples:
- "Showing mitochondria 123 with volume 3.05e11 nm³"
- "Displaying the 3 largest mitochondria (volumes: 5.2e11 nm³, 4.8e11 nm³, 4.3e11 nm³)"
- "Showing 5 nuclei with volumes ranging from 1.2e12 to 2.5e12 nm³"

Response:"""

        try:
            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] VISUALIZATION ANSWER", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["message"]["content"].strip()

            if self.verbose:
                print("RESPONSE:", flush=True)
                print(answer, flush=True)
                print("="*80 + "\n", flush=True)

            # Track AI interaction
            if ai_interactions is not None:
                ai_interactions.append({
                    'type': 'visualization_answer',
                    'prompt': prompt,
                    'response': answer,
                    'model': self.model
                })

            print(f"[QUERY_AGENT] AI formatted visualization answer", flush=True)
            return answer

        except Exception as e:
            print(f"[QUERY_AGENT] Error formatting visualization answer with AI: {e}, using fallback", flush=True)
            # Fallback to simple format
            count = len(segment_ids)
            type_name = organelle_type.replace('_', ' ') if organelle_type else "organelle"
            if count == 1:
                vol = results[0].get('volume')
                vol_str = f" with volume {vol:.2e} nm³" if vol else ""
                return f"Showing {type_name} {segment_ids[0]}{vol_str}"
            else:
                return f"Showing {count} {type_name} objects"

    def _get_voxel_size_from_state(self) -> tuple:
        """
        Extract voxel size from Neuroglancer state.

        Returns:
            Tuple of (voxel_size_x, voxel_size_y, voxel_size_z) in nanometers
        """
        if not self.ng_tracker:
            print(f"[QUERY_AGENT] No NG tracker, using default voxel size (8, 8, 8)", flush=True)
            return (8, 8, 8)

        try:
            # Get the full state JSON
            with self.ng_tracker.viewer.txn() as s:
                state = s.to_json()

            # Debug: print dimensions structure
            if "dimensions" in state:
                print(f"[QUERY_AGENT] State dimensions: {state['dimensions']}", flush=True)

                # Neuroglancer dimensions format: {axis_name: [scale, unit], ...}
                # e.g., {'x': [6e-09, 'm'], 'y': [6e-09, 'm'], 'z': [6e-09, 'm']}
                # The scale is in meters, we need to convert to nanometers (multiply by 1e9)
                dims = state["dimensions"]
                if isinstance(dims, dict):
                    voxel_sizes = []
                    for axis in ['x', 'y', 'z']:
                        if axis in dims:
                            dim_info = dims[axis]
                            if isinstance(dim_info, list) and len(dim_info) >= 2:
                                # First element is the scale, second is the unit
                                scale = float(dim_info[0])
                                unit = dim_info[1] if len(dim_info) > 1 else 'm'

                                # Convert to nanometers
                                if unit == 'm':
                                    # Meters to nanometers: multiply by 1e9
                                    voxel_size_nm = scale * 1e9
                                elif unit == 'nm':
                                    voxel_size_nm = scale
                                else:
                                    print(f"[QUERY_AGENT] Unknown unit '{unit}', assuming meters", flush=True)
                                    voxel_size_nm = scale * 1e9

                                voxel_sizes.append(voxel_size_nm)
                            else:
                                voxel_sizes.append(1.0)
                        else:
                            voxel_sizes.append(1.0)

                    if len(voxel_sizes) == 3:
                        result = tuple(voxel_sizes)
                        print(f"[QUERY_AGENT] Extracted voxel size (in nm): {result}", flush=True)
                        return result

            print(f"[QUERY_AGENT] Could not parse voxel size, using default (8, 8, 8)", flush=True)
            return (8, 8, 8)

        except Exception as e:
            print(f"[QUERY_AGENT] Error getting voxel size from state: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return (8, 8, 8)

    def _nm_to_voxel_coords(self, x_nm: float, y_nm: float, z_nm: float) -> List[int]:
        """
        Convert nanometer coordinates to voxel coordinates.

        The database stores positions in nanometers (from CSV 'COM X/Y/Z (nm)' columns),
        but Neuroglancer state uses voxel coordinates. We need to divide by the voxel size.

        Args:
            x_nm, y_nm, z_nm: Coordinates in nanometers

        Returns:
            List of [x_voxel, y_voxel, z_voxel] as integers
        """
        # Get voxel size from Neuroglancer state
        voxel_size = self._get_voxel_size_from_state()

        # Convert nm to voxels by dividing by voxel size
        x_voxel = int(x_nm / voxel_size[0])
        y_voxel = int(y_nm / voxel_size[1])
        z_voxel = int(z_nm / voxel_size[2])

        print(f"[QUERY_AGENT] Converted ({x_nm:.1f}, {y_nm:.1f}, {z_nm:.1f}) nm → ({x_voxel}, {y_voxel}, {z_voxel}) voxels (voxel_size={voxel_size})", flush=True)

        return [x_voxel, y_voxel, z_voxel]

    def _calculate_zoom_for_volume(self, volume: float) -> int:
        """
        Calculate appropriate zoom level (crossSectionScale) for object volume.

        Larger objects need higher scale (zoomed out).
        Smaller objects need lower scale (zoomed in).

        Args:
            volume: Object volume (can be None)

        Returns:
            Appropriate scale value
        """
        # Handle None or invalid volume
        if volume is None or volume <= 0:
            return 100  # Default medium zoom

        if volume < 1000:
            return 10  # Very zoomed in for tiny objects
        elif volume < 10000:
            return 30
        elif volume < 100000:
            return 100
        elif volume < 500000:
            return 200
        else:
            return 300  # Zoomed out for large objects

    def _format_answer(
        self,
        results: List[Dict[str, Any]],
        user_query: str,
        ai_interactions: List[Dict[str, Any]] = None
    ) -> str:
        """
        Format query results into natural language answer using LLM.

        Args:
            results: Query results
            user_query: Original query

        Returns:
            Natural language answer
        """
        if not results:
            return "No results found."

        # If single aggregate value (COUNT, AVG, etc.) - use simple formatting
        if len(results) == 1 and len(results[0]) == 1:
            key = list(results[0].keys())[0]
            value = results[0][key]

            # Infer units from key name
            unit = ""
            if 'volume' in key.lower():
                unit = " nm³"
            elif 'surface' in key.lower() or 'area' in key.lower():
                unit = " nm²"
            elif 'position' in key.lower() or 'lsp' in key.lower() or 'radius' in key.lower() or 'length' in key.lower():
                unit = " nm"

            if key == 'count' or 'count' in key.lower():
                return f"There are {int(value)} matching organelles."
            elif 'average' in key.lower() or 'avg' in key.lower():
                if value >= 1e6:
                    return f"The average is {value:.2e}{unit}"
                else:
                    return f"The average is {value:.2f}{unit}"
            elif 'sum' in key.lower() or 'total' in key.lower():
                if value >= 1e6:
                    return f"The total is {value:.2e}{unit}"
                else:
                    return f"The total is {value:.2f}{unit}"
            elif 'min' in key.lower():
                if value >= 1e6:
                    return f"The minimum is {value:.2e}{unit}"
                else:
                    return f"The minimum is {value:.2f}{unit}"
            elif 'max' in key.lower():
                if value >= 1e6:
                    return f"The maximum is {value:.2e}{unit}"
                else:
                    return f"The maximum is {value:.2f}{unit}"
            else:
                if value >= 1e6:
                    return f"The {key} is {value:.2e}{unit}"
                else:
                    return f"The {key} is {value}{unit}"

        # For complex results, use LLM to format answer
        return self._format_answer_with_llm(results, user_query, ai_interactions)

    def _format_answer_with_llm(
        self,
        results: List[Dict[str, Any]],
        user_query: str,
        ai_interactions: List[Dict[str, Any]] = None
    ) -> str:
        """
        Use LLM (nemotron-3-nano) to format results into a natural language answer.

        Args:
            results: Query results
            user_query: Original query
            ai_interactions: Optional list to track AI interactions

        Returns:
            Formatted natural language answer
        """
        # Limit results shown to LLM to avoid token limits
        max_results = 10
        results_to_show = results[:max_results]

        # Format results as JSON for the LLM
        import json
        results_json = json.dumps(results_to_show, indent=2)

        prompt = f"""You are answering a user's question about organelle data from microscopy analysis.

User Question: {user_query}

Query Results:
{results_json}

Instructions:
1. Answer the user's question directly and naturally
2. If there are multiple results, describe ALL of them clearly
3. Format large numbers in scientific notation (e.g., 3.05e11) for readability
4. Be concise but informative - include key details like volume and surface area
5. Keep your answer to 2-3 sentences maximum
6. Use natural language, not just listing data
7. ALWAYS include units in your answer:
   - Volume: (nm³) - cubic nanometers
   - Surface area: (nm²) - square nanometers
   - Position coordinates: (nm) - nanometers
   - Lengths/distances: (nm) - nanometers
   - Example: "The volume is 3.05e11 nm³" or "Position at (1234, 5678, 9012) nm"
8. NOTE: Position coordinates in the database are in (position_x, position_y, position_z) order
   which corresponds to the order used in the CSV files and Neuroglancer viewer.

Answer:"""

        try:
            if self.verbose:
                print("\n" + "="*80, flush=True)
                print("[AI] ANSWER FORMATTING", flush=True)
                print("="*80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("-"*80, flush=True)

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["message"]["content"].strip()

            if self.verbose:
                print("RESPONSE:", flush=True)
                print(answer, flush=True)
                print("="*80 + "\n", flush=True)

            # Track AI interaction
            if ai_interactions is not None:
                ai_interactions.append({
                    'type': 'answer_formatting',
                    'prompt': prompt,
                    'response': answer,
                    'model': self.model
                })

            print(f"[QUERY_AGENT] LLM formatted answer: {answer}", flush=True)
            return answer

        except Exception as e:
            print(f"[QUERY_AGENT] Error formatting answer with LLM: {e}", flush=True)
            # Fallback to simple formatting
            if len(results) == 1:
                return f"Found 1 result: {results[0].get('object_id', 'unknown')}"
            else:
                return f"Found {len(results)} results."
