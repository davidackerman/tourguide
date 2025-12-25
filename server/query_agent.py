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

    def __init__(self, db: OrganelleDatabase, model: str = "nemotron"):
        """
        Initialize query agent.

        Args:
            db: OrganelleDatabase instance
            model: Ollama model name (default: nemotron)
        """
        self.db = db
        self.model = model
        print(f"[QUERY_AGENT] Initialized with model: {model}", flush=True)

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process natural language query and return results.

        Args:
            user_query: Natural language query from user

        Returns:
            Dictionary with query results and metadata
        """
        if not user_query.strip():
            return {
                "type": "error",
                "answer": "Please provide a query."
            }

        try:
            # Classify query type
            query_type = self._classify_query_type(user_query)

            # Generate SQL
            sql = self._generate_sql(user_query)

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

            # Execute query
            results = self.db.execute_query(sql)

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
            if query_type == "navigation":
                return self._handle_navigation_query(user_query, sql, results)
            else:
                return self._handle_informational_query(user_query, sql, results)

        except Exception as e:
            print(f"[QUERY_AGENT] Error processing query: {e}", flush=True)
            return {
                "type": "error",
                "answer": f"Error processing query: {str(e)}"
            }

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query as navigational or informational.

        Args:
            query: User query string

        Returns:
            'navigation' or 'informational'
        """
        query_lower = query.lower()

        for keyword in self.NAVIGATION_KEYWORDS:
            if keyword in query_lower:
                return "navigation"

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

        prompt = f"""You are an expert at converting natural language to SQLite queries.

Database Schema:
{schema_desc}

User Query: {user_query}

Generate ONLY a valid SQLite SELECT query. No explanations, no markdown formatting, just the SQL.
Query must be safe (no DROP, INSERT, UPDATE, DELETE).
Use proper SQLite syntax.

Example Queries:
- "What is the size of the biggest mito?"
  → SELECT object_id, volume FROM organelles WHERE organelle_type='mitochondria' ORDER BY volume DESC LIMIT 1;

- "How many nuclei are there?"
  → SELECT COUNT(*) as count FROM organelles WHERE organelle_type='nucleus';

- "Take me to the biggest nucleus"
  → SELECT object_id, volume, position_x, position_y, position_z FROM organelles WHERE organelle_type='nucleus' ORDER BY volume DESC LIMIT 1;

- "What is the average volume of ER?"
  → SELECT AVG(volume) as average_volume FROM organelles WHERE organelle_type='endoplasmic_reticulum';

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
        Format query results into natural language answer.

        Args:
            results: Query results
            user_query: Original query

        Returns:
            Natural language answer
        """
        if not results:
            return "No results found."

        # If single value (COUNT, AVG, etc.)
        if len(results) == 1 and len(results[0]) == 1:
            key = list(results[0].keys())[0]
            value = results[0][key]

            if key == 'count':
                return f"There are {value} matching organelles."
            elif 'average' in key.lower() or 'avg' in key.lower():
                return f"The average is {value:.2f}"
            else:
                return f"The {key} is {value}"

        # If asking about "biggest" or "largest"
        if 'biggest' in user_query.lower() or 'largest' in user_query.lower():
            result = results[0]
            object_id = result.get('object_id', 'unknown')
            volume = result.get('volume')
            organelle_type = result.get('organelle_type', 'organelle')

            if volume is not None:
                return f"The biggest {organelle_type} is {object_id} with volume {volume:.1f}"
            else:
                return f"The biggest {organelle_type} is {object_id}"

        # If asking about "smallest"
        if 'smallest' in user_query.lower():
            result = results[0]
            object_id = result.get('object_id', 'unknown')
            volume = result.get('volume')
            organelle_type = result.get('organelle_type', 'organelle')

            if volume is not None:
                return f"The smallest {organelle_type} is {object_id} with volume {volume:.1f}"
            else:
                return f"The smallest {organelle_type} is {object_id}"

        # If multiple results, summarize
        if len(results) > 5:
            return f"Found {len(results)} results."

        # Otherwise, format first few results
        formatted_results = []
        for result in results[:3]:
            object_id = result.get('object_id', 'unknown')
            volume = result.get('volume')
            if volume is not None:
                formatted_results.append(f"{object_id} (volume: {volume:.1f})")
            else:
                formatted_results.append(object_id)

        answer = "Results: " + ", ".join(formatted_results)
        if len(results) > 3:
            answer += f" and {len(results) - 3} more"

        return answer
