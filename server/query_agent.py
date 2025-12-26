"""
Query Agent for Natural Language to SQL

Uses Ollama/Nemotron to convert user queries to SQL and process results.
"""

import re
from typing import Dict, Any, List, Optional
import ollama
from organelle_db import OrganelleDatabase


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

    def __init__(self, db: OrganelleDatabase, model: str = "nemotron", ng_tracker=None):
        """
        Initialize query agent.

        Args:
            db: OrganelleDatabase instance
            model: Ollama model name (default: nemotron)
            ng_tracker: Optional NG_StateTracker instance for layer discovery
        """
        self.db = db
        self.model = model
        self.ng_tracker = ng_tracker

        # Discover available layers at init
        self.available_layers = {}
        if ng_tracker:
            self.available_layers = ng_tracker.get_available_layers()
            print(f"[QUERY_AGENT] Discovered layers: {list(self.available_layers.keys())}", flush=True)

        print(f"[QUERY_AGENT] Initialized with model: {model}", flush=True)

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process natural language query and return results.

        Args:
            user_query: Natural language query from user

        Returns:
            Dictionary with query results and metadata
        """
        import time

        timing = {}
        start_total = time.time()

        if not user_query.strip():
            return {
                "type": "error",
                "answer": "Please provide a query."
            }

        try:
            # Classify query type
            query_type = self._classify_query_type(user_query)

            # Generate SQL with timing
            start_sql = time.time()
            sql = self._generate_sql(user_query)
            timing['sql_generation'] = time.time() - start_sql

            if not sql:
                return {
                    "type": "error",
                    "answer": "Could not generate a valid query. Try asking something like: 'What is the size of the biggest mito?'"
                }

            # Validate SQL
            if not self._validate_sql(sql):
                return {
                    "type": "error",
                    "answer": "Generated query contains unsafe operations. Please try rephrasing."
                }

            # Execute query with timing
            start_exec = time.time()
            results = self.db.execute_query(sql)
            timing['query_execution'] = time.time() - start_exec

            # Handle empty results
            if not results:
                available_types = self.db.get_available_organelle_types()
                return {
                    "type": "error",
                    "answer": f"No results found. Available organelle types: {', '.join(available_types)}. Try asking about one of these.",
                    "available_types": available_types,
                    "sql": sql
                }

            # Format response based on query type
            start_format = time.time()

            # Smart fallback: if query type is visualization but we have position data
            # and only 1 result, it's likely a navigation query
            if query_type == "visualization" and len(results) == 1:
                first_result = results[0]
                has_position = all(k in first_result for k in ['position_x', 'position_y', 'position_z'])
                if has_position and all(first_result[k] is not None for k in ['position_x', 'position_y', 'position_z']):
                    print("[QUERY_AGENT] Auto-correcting to navigation (has position data, single result)", flush=True)
                    query_type = "navigation"

            if query_type == "navigation":
                result = self._handle_navigation_query(user_query, sql, results)
            elif query_type == "visualization":
                result = self._handle_visualization_query(user_query, sql, results)
            else:
                result = self._handle_informational_query(user_query, sql, results)
            timing['answer_formatting'] = time.time() - start_format

            # Add timing info to result
            timing['total'] = time.time() - start_total
            result['timing'] = timing

            return result

        except Exception as e:
            print(f"[QUERY_AGENT] Error processing query: {e}", flush=True)
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
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            intent = response["message"]["content"].strip().lower()

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

    def _generate_sql(self, user_query: str) -> Optional[str]:
        """
        Generate SQL query from natural language using Ollama.

        Args:
            user_query: Natural language query

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

        prompt = f"""You are an expert at converting natural language to SQLite queries.

Database Schema:
{schema_desc}

{fuzzy_matching_note}

User Query: {user_query}

IMPORTANT RULES:
1. Generate ONLY a valid, complete SQLite SELECT query
2. The query MUST include FROM organelles
3. The query MUST end with a semicolon
4. No explanations, no markdown formatting, just the SQL
5. Query must be safe (no DROP, INSERT, UPDATE, DELETE)
6. Use proper SQLite syntax
7. Match user's organelle names to the available types listed above

Example Queries:
- "What is the size of the biggest mito?"
  → SELECT object_id, volume FROM organelles WHERE organelle_type='mitochondria' ORDER BY volume DESC LIMIT 1;

- "How many nuclei are there?"
  → SELECT COUNT(*) as count FROM organelles WHERE organelle_type='nucleus';

- "Take me to the biggest nucleus"
  → SELECT object_id, volume, position_x, position_y, position_z FROM organelles WHERE organelle_type='nucleus' ORDER BY volume DESC LIMIT 1;

- "What is the average volume of ER?"
  → SELECT AVG(volume) as average_volume FROM organelles WHERE organelle_type='endoplasmic_reticulum';

- "Describe the top 3 largest nuclei"
  → SELECT object_id, volume, surface_area, position_x, position_y, position_z FROM organelles WHERE organelle_type='nucleus' ORDER BY volume DESC LIMIT 3;

- "Show only the 3 largest mitos"
  → SELECT object_id, organelle_type, volume FROM organelles WHERE organelle_type='mitochondria' ORDER BY volume DESC LIMIT 3;

IMPORTANT:
- Always include object_id and organelle_type in SELECT for queries that filter or list specific organelles
- For navigation queries (take me to, go to, etc.), MUST include position_x, position_y, position_z

Remember: Your response must be ONLY the SQL query, nothing else.

SQL Query:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            sql = response["message"]["content"].strip()

            # Clean up response (remove markdown, extra text)
            sql = self._clean_sql(sql)

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

        # If multiple lines, take the line that looks like SQL
        lines = sql.split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT'):
                sql = line
                break

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
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
            layer_name = response["message"]["content"].strip()

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
