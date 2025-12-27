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
            ng_tracker: Optional NG_StateTracker instance for layer discovery
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

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query intent using AI.

        Args:
            query: User query string

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

    def _generate_sql(self, user_query: str, previous_error: Optional[str] = None) -> Optional[str]:
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

    def _generate_sql_with_retry(self, user_query: str, max_retries: int = 2) -> Optional[str]:
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
            sql = self._generate_sql(user_query, previous_error=last_error if attempt > 0 else None)

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

        try:
            # Classify query type
            query_type = self._classify_query_type(user_query)

            # Generate SQL with retry mechanism and timing
            start_sql = time.time()
            sql = self._generate_sql_with_retry(user_query, max_retries=2)
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
            if query_type == "informational":
                return self._handle_informational_query(user_query, sql, results)
            elif query_type == "navigation":
                return self._handle_navigation_query(user_query, sql, results)
            elif query_type == "visualization":
                return self._handle_visualization_query(user_query, sql, results)
            else:
                # Default to informational
                return self._handle_informational_query(user_query, sql, results)

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
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle informational query (no navigation).

        Args:
            user_query: Original user query
            sql: Generated SQL
            results: Query results

        Returns:
            Response dictionary
        """
        # Format answer
        answer = self._format_answer(results, user_query)

        return {
            "type": "informational",
            "sql": sql,
            "results": results,
            "answer": answer
        }

    def _handle_navigation_query(
        self,
        user_query: str,
        sql: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle navigation query (update viewer position).

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

        # Calculate zoom level
        volume = result.get('volume', 0)
        scale = self._calculate_zoom_for_volume(volume)

        # Create navigation command
        navigation = {
            "position": [int(position_x), int(position_y), int(position_z)],
            "object_id": result.get('object_id', 'unknown'),
            "scale": scale
        }

        # Format answer
        object_id = result.get('object_id', 'unknown')
        organelle_type = result.get('organelle_type', 'organelle')
        volume_str = f" with volume {volume:.1f}" if volume else ""

        answer = f"Taking you to {organelle_type} {object_id}{volume_str}"

        return {
            "type": "navigation",
            "sql": sql,
            "results": results,
            "answer": answer,
            "navigation": navigation
        }

    def _handle_visualization_query(
        self,
        user_query: str,
        sql: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle visualization query (show/hide specific segments).

        Args:
            user_query: Original user query
            sql: Generated SQL
            results: Query results

        Returns:
            Response dictionary with visualization commands
        """
        # Extract segment IDs and organelle type from results
        segment_ids = []
        organelle_type = None

        for result in results:
            obj_id = result.get('object_id')
            if obj_id:
                segment_ids.append(str(obj_id))
            if not organelle_type and result.get('organelle_type'):
                organelle_type = result.get('organelle_type')

        if not segment_ids:
            return {
                "type": "error",
                "answer": "No objects found to visualize.",
                "sql": sql,
                "results": results
            }

        # Map organelle type to layer name
        layer_name = self._get_layer_name(organelle_type)

        # Create visualization command
        visualization = {
            "layer_name": layer_name,
            "segment_ids": segment_ids,
            "action": "show_only"  # show_only, add, remove
        }

        # Format answer using AI
        answer = self._format_visualization_answer(user_query, results, organelle_type, segment_ids)

        return {
            "type": "visualization",
            "sql": sql,
            "results": results,
            "answer": answer,
            "visualization": visualization
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
        segment_ids: List[str]
    ) -> str:
        """
        Format a natural language answer for visualization query.

        Args:
            user_query: Original user query
            results: Query results
            organelle_type: Type of organelle (can be None)
            segment_ids: List of segment IDs to show

        Returns:
            Natural language answer
        """
        count = len(segment_ids)
        type_name = organelle_type.replace('_', ' ') if organelle_type else "organelle"

        # Simple format for now - could use LLM for more natural responses
        if count == 1:
            vol = results[0].get('volume')
            vol_str = f" with volume {vol:.2e}" if vol else ""
            return f"Showing {type_name} {segment_ids[0]}{vol_str}"
        else:
            return f"Showing {count} {type_name} objects"

    def _calculate_zoom_for_volume(self, volume: float) -> int:
        """
        Calculate appropriate zoom level (crossSectionScale) for object volume.

        Larger objects need higher scale (zoomed out).
        Smaller objects need lower scale (zoomed in).

        Args:
            volume: Object volume

        Returns:
            Appropriate scale value
        """
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
        user_query: str
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

            if key == 'count' or 'count' in key.lower():
                return f"There are {int(value)} matching organelles."
            elif 'average' in key.lower() or 'avg' in key.lower():
                return f"The average is {value:.2f}"
            elif 'sum' in key.lower() or 'total' in key.lower():
                return f"The total is {value:.2f}"
            elif 'min' in key.lower():
                return f"The minimum is {value:.2f}"
            elif 'max' in key.lower():
                return f"The maximum is {value:.2f}"
            else:
                return f"The {key} is {value}"

        # For complex results, use LLM to format answer
        return self._format_answer_with_llm(results, user_query)

    def _format_answer_with_llm(
        self,
        results: List[Dict[str, Any]],
        user_query: str
    ) -> str:
        """
        Use LLM (nemotron-3-nano) to format results into a natural language answer.

        Args:
            results: Query results
            user_query: Original query

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

Answer:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["message"]["content"].strip()
            print(f"[QUERY_AGENT] LLM formatted answer: {answer}", flush=True)
            return answer

        except Exception as e:
            print(f"[QUERY_AGENT] Error formatting answer with LLM: {e}", flush=True)
            # Fallback to simple formatting
            if len(results) == 1:
                return f"Found 1 result: {results[0].get('object_id', 'unknown')}"
            else:
                return f"Found {len(results)} results."
