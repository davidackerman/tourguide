"""
Analysis Agent for Data Analysis Code Generation.

Uses AI (Gemini/Claude/Ollama) to convert natural language queries
into Python code for data analysis and visualization.
"""

import re
import os
from typing import Dict, Any, Optional, Tuple
from organelle_db import OrganelleDatabase


class AnalysisAgent:
    """AI-powered agent that generates data analysis code."""

    # Whitelist of allowed imports
    ALLOWED_IMPORTS = {
        'sqlite3', 'pandas', 'numpy', 'plotly', 'matplotlib',
        'seaborn', 'json', 'math', 'statistics', 'sys'
    }

    # Forbidden patterns in code
    FORBIDDEN_PATTERNS = [
        (r'\bos\.', 'os module usage'),
        (r'\bsubprocess\.', 'subprocess module usage'),
        (r'\beval\(', 'eval() function'),
        (r'\bexec\(', 'exec() function'),
        (r'__import__', '__import__ function'),
        (r'\brequests\.', 'requests module usage'),
        (r'\bsocket\.', 'socket module usage'),
        (r'\bpickle\.', 'pickle module usage'),
        (r'\bshutil\.', 'shutil module usage'),
        (r'open\([^\'\"]*[\'\"]/(?!output)', 'file operations outside /output'),
    ]

    def __init__(self, db: OrganelleDatabase, provider: Optional[str] = None):
        """
        Initialize analysis agent.

        Args:
            db: OrganelleDatabase instance
            provider: AI provider ('gemini', 'claude', 'local', or None for auto-detect)
        """
        self.db = db

        # Check for analysis-specific provider override
        analysis_provider = os.environ.get("ANALYSIS_AI_PROVIDER", "").lower()

        # Auto-detect provider based on available API keys and ANALYSIS_AI_MODEL
        self.provider = provider or analysis_provider or os.environ.get("AI_PROVIDER", "auto")
        self.use_local = os.environ.get("USE_LOCAL", "false").lower() == "true"

        # If ANALYSIS_AI_MODEL is set to an Ollama model, force local provider
        analysis_model = os.environ.get("ANALYSIS_AI_MODEL", "")
        ollama_models = ["qwen", "nemotron", "llama", "codellama", "deepseek", "mistral", "phi"]
        if analysis_model and any(model in analysis_model.lower() for model in ollama_models):
            self.provider = "local"
            print(f"[ANALYSIS_AGENT] Detected Ollama model '{analysis_model}', using local provider")
        elif self.provider == "auto":
            if self.use_local:
                self.provider = "local"
            elif os.environ.get("GOOGLE_API_KEY"):
                self.provider = "gemini"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.provider = "claude"
            else:
                print("[ANALYSIS_AGENT] WARNING: No API key found. Set USE_LOCAL=true or provide API key.")
                self.enabled = False
                return

        # Initialize the appropriate client
        if self.provider == "local":
            try:
                import ollama
                ollama.list()  # Test connection
                self.client = ollama
                # Use ANALYSIS_AI_MODEL if set, otherwise fall back to QUERY_AI_MODEL
                self.model = os.environ.get("ANALYSIS_AI_MODEL") or os.environ.get("QUERY_AI_MODEL", "qwen2.5-coder:1.5b")
                self.enabled = True
                print(f"[ANALYSIS_AGENT] Using Local Mode (Ollama {self.model})")
            except Exception as e:
                print(f"[ANALYSIS_AGENT] ERROR: Failed to connect to Ollama: {e}")
                self.enabled = False
                return

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                # Use ANALYSIS_AI_MODEL if set for Gemini, otherwise use GEMINI_MODEL
                model_name = os.environ.get("ANALYSIS_AI_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-002")
                self.client = genai.GenerativeModel(model_name)
                self.enabled = True
                print(f"[ANALYSIS_AGENT] Using Gemini model: {model_name}")
            except Exception as e:
                print(f"[ANALYSIS_AGENT] ERROR: Failed to initialize Gemini: {e}")
                self.enabled = False
                return

        elif self.provider == "claude":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                # Use ANALYSIS_AI_MODEL if set, otherwise default to Claude 3.5 Sonnet
                self.model = os.environ.get("ANALYSIS_AI_MODEL", "claude-3-5-sonnet-20241022")
                self.enabled = True
                print(f"[ANALYSIS_AGENT] Using Claude model: {self.model}")
            except Exception as e:
                print(f"[ANALYSIS_AGENT] ERROR: Failed to initialize Claude: {e}")
                self.enabled = False
                return

        print(f"[ANALYSIS_AGENT] Initialized with provider: {self.provider}")

    def generate_analysis_code(self, user_query: str) -> Dict[str, Any]:
        """
        Generate Python analysis code from natural language query.

        Args:
            user_query: Natural language query from user

        Returns:
            Dictionary with 'code' (str) and optional 'error' (str)
        """
        if not self.enabled:
            return {
                "error": "Analysis agent not enabled. Configure AI provider."
            }

        if not user_query.strip():
            return {
                "error": "Please provide a query."
            }

        try:
            # Build prompt with schema and examples
            prompt = self._build_analysis_prompt(user_query)

            # Call AI to generate code
            code = self._call_ai(prompt)

            if not code:
                return {
                    "error": "Failed to generate code from AI."
                }

            # Validate code safety
            is_safe, error_msg = self._validate_code_safety(code)
            if not is_safe:
                print(f"[ANALYSIS_AGENT] Unsafe code detected: {error_msg}")
                return {
                    "error": f"Generated code contains unsafe operations: {error_msg}"
                }

            print(f"[ANALYSIS_AGENT] Generated safe code ({len(code)} chars)")
            return {
                "code": code,
                "language": "python"
            }

        except Exception as e:
            print(f"[ANALYSIS_AGENT] Error generating code: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Code generation failed: {str(e)}"
            }

    def _build_analysis_prompt(self, user_query: str) -> str:
        """
        Build comprehensive prompt for code generation.

        Args:
            user_query: User's natural language query

        Returns:
            Formatted prompt string
        """
        # Get database schema and available types
        schema_desc = self.db.get_schema_description()
        available_types = self.db.get_available_organelle_types()

        prompt = f"""You are an expert Python data analyst. Generate Python code to answer the user's question about organelle data.

DATABASE SCHEMA:
{schema_desc}

AVAILABLE ORGANELLE TYPES:
{', '.join(available_types)}

USER QUERY: {user_query}

CODE REQUIREMENTS:
1. Use ONLY these libraries: pandas, numpy, plotly, matplotlib, seaborn, sqlite3, json, math, statistics, sys
2. Connect to database at '/data/organelles.db'
3. Save ALL plots to '/output/' directory:
   - For plotly: use fig.write_html('/output/plot.html')
   - For matplotlib: use plt.savefig('/output/plot.png', dpi=100, bbox_inches='tight')
4. Print statistics/results to stdout (they will be displayed to user)
5. Always close database connection (use 'with' context manager or try/except/finally)
6. Use proper units in output: nm³ for volume, nm² for surface area, nm for distances
7. Handle errors gracefully with try/except or try/except/finally (NEVER use try/finally without except)
8. NEVER use dtype parameter in pd.read_sql_query() - let pandas infer types automatically
9. Use pd.to_numeric() AFTER reading data if you need to ensure numeric types
10. IMPORTANT: If using try/finally, you MUST include except: try/except/finally or just use 'with' statement

FORBIDDEN:
- NO import os, subprocess, requests, socket, pickle, eval, exec
- NO file operations outside /output/
- NO network access
- NO system calls

CODE EXAMPLES:

Example 1: Histogram of volumes
```python
import sqlite3
import pandas as pd
import plotly.express as px

with sqlite3.connect('/data/organelles.db') as conn:
    df = pd.read_sql_query("SELECT volume FROM organelles WHERE organelle_type='mitochondria'", conn)

fig = px.histogram(df, x='volume', title='Mitochondria Volume Distribution',
                   labels={{'volume': 'Volume (nm³)'}})
fig.write_html('/output/plot.html')

print(f"Total: {{len(df)}} mitochondria")
print(f"Mean volume: {{df['volume'].mean():.2e}} nm³")
print(f"Median volume: {{df['volume'].median():.2e}} nm³")
```

Example 2: Statistics by organelle type
```python
import sqlite3
import pandas as pd

with sqlite3.connect('/data/organelles.db') as conn:
    df = pd.read_sql_query("SELECT organelle_type, volume, surface_area FROM organelles", conn)

stats = df.groupby('organelle_type').agg({{'volume': ['mean', 'count'], 'surface_area': 'mean'}})

print("Statistics by Organelle Type:")
print(stats.to_string())
```

Example 3: Scatter plot with matplotlib
```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

with sqlite3.connect('/data/organelles.db') as conn:
    df = pd.read_sql_query("SELECT volume, surface_area, organelle_type FROM organelles", conn)

plt.figure(figsize=(10, 6))
for org_type in df['organelle_type'].unique():
    subset = df[df['organelle_type'] == org_type]
    plt.scatter(subset['volume'], subset['surface_area'], label=org_type, alpha=0.6)

plt.xlabel('Volume (nm³)')
plt.ylabel('Surface Area (nm²)')
plt.title('Volume vs Surface Area by Organelle Type')
plt.legend()
plt.tight_layout()
plt.savefig('/output/plot.png', dpi=100, bbox_inches='tight')
plt.close()

print("Scatter plot saved")
```

Example 4: Box plot with seaborn
```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with sqlite3.connect('/data/organelles.db') as conn:
    df = pd.read_sql_query("SELECT organelle_type, volume FROM organelles", conn)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='organelle_type', y='volume')
plt.xticks(rotation=45)
plt.ylabel('Volume (nm³)')
plt.title('Volume Distribution by Organelle Type')
plt.tight_layout()
plt.savefig('/output/plot.png', dpi=100, bbox_inches='tight')
plt.close()

print(f"Box plot created for {{df['organelle_type'].nunique()}} organelle types")
```

Example 5: Multiple plots with plotly
```python
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

with sqlite3.connect('/data/organelles.db') as conn:
    df = pd.read_sql_query("SELECT organelle_type, volume, surface_area FROM organelles", conn)

# Create subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Volume Distribution', 'Surface Area Distribution'))

for org_type in df['organelle_type'].unique()[:3]:  # Limit to 3 types for clarity
    subset = df[df['organelle_type'] == org_type]
    fig.add_trace(go.Box(y=subset['volume'], name=org_type), row=1, col=1)
    fig.add_trace(go.Box(y=subset['surface_area'], name=org_type, showlegend=False), row=1, col=2)

fig.update_xaxes(title_text="Organelle Type", row=1, col=1)
fig.update_xaxes(title_text="Organelle Type", row=1, col=2)
fig.update_yaxes(title_text="Volume (nm³)", row=1, col=1)
fig.update_yaxes(title_text="Surface Area (nm²)", row=1, col=2)
fig.update_layout(title_text="Organelle Metrics Comparison")

fig.write_html('/output/plot.html')

print(f"Interactive comparison plot created")
```

CRITICAL REQUIREMENTS:
1. Output ONLY valid Python code - nothing else
2. NO explanations, NO comments about what you're doing, NO reasoning
3. NO thinking tags like <think>, <reasoning>, </think>, etc.
4. NO natural language between code sections
5. Code must be syntactically correct and executable
6. Always save visualizations to /output/
7. Print meaningful statistics to help answer the user's question
8. Use scientific notation for large numbers

Your response should start with 'import' and contain ONLY Python code.

Generate the Python code:"""

        return prompt

    def _call_ai(self, prompt: str) -> Optional[str]:
        """
        Call AI provider to generate code.

        Args:
            prompt: Formatted prompt

        Returns:
            Generated Python code or None if failed
        """
        try:
            if self.provider == "local":
                # Ollama
                import ollama
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                code = response["message"]["content"].strip()

            elif self.provider == "gemini":
                # Gemini
                response = self.client.generate_content(prompt)
                code = response.text.strip()

            elif self.provider == "claude":
                # Claude
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                code = response.content[0].text.strip()

            else:
                print(f"[ANALYSIS_AGENT] Unknown provider: {self.provider}")
                return None

            # Clean code (remove markdown formatting)
            code = self._clean_code(code)

            return code

        except Exception as e:
            print(f"[ANALYSIS_AGENT] AI call failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _clean_code(self, code: str) -> str:
        """
        Clean AI-generated code by removing markdown formatting and reasoning artifacts.

        Args:
            code: Raw code from AI

        Returns:
            Cleaned Python code
        """
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)

        # Remove thinking/reasoning tags that some models output
        # Match opening tags like <think>, <reasoning>, etc.
        code = re.sub(r'</?(?:think|thinking|reasoning|reason|thought)(?:\s[^>]*)?>.*?(?=\n|$)', '', code, flags=re.IGNORECASE)

        # Remove closing tags that might be on their own line
        code = re.sub(r'^.*?</(?:think|thinking|reasoning|reason|thought)>.*?$', '', code, flags=re.MULTILINE | re.IGNORECASE)

        # Remove lines that look like natural language explanations between code sections
        # (lines that start without indentation and don't look like Python)
        lines = code.split('\n')
        cleaned_lines = []
        in_reasoning = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines but preserve them in output
            if not stripped:
                if not in_reasoning:
                    cleaned_lines.append(line)
                continue

            # Detect reasoning/thinking tags
            if re.match(r'</?(?:think|thinking|reasoning|reason|thought)', stripped, re.IGNORECASE):
                in_reasoning = '</' not in stripped  # Exit reasoning when we hit closing tag
                continue

            # Skip lines that look like natural language (not code)
            # Heuristic: If line doesn't start with common Python patterns and has prose-like text
            if not in_reasoning:
                # Check if line looks like Python code
                is_code = (
                    stripped.startswith(('#', 'import ', 'from ', 'def ', 'class ', 'if ',
                                       'for ', 'while ', 'try:', 'except', 'finally:',
                                       'with ', 'return ', 'raise ', 'assert ', 'yield ',
                                       'print(', 'pass', '}', '{', ']', '[', ')')) or
                    '=' in line or  # Assignment
                    line.startswith(' ') or  # Indented code
                    line.startswith('\t') or  # Indented code
                    stripped.endswith((':', ',', ')', ']', '}')) or  # Code continuation
                    re.match(r'^\w+\(', stripped) or  # Function call
                    re.match(r'^\w+\.', stripped)  # Method call like plt.close()
                )

                # If it looks like prose (contains explanation patterns), skip it
                is_prose = bool(re.match(
                    r'^(?:We need|Let\'s|This will|Now|First|Next|Then|Note:|Important:|Here)',
                    stripped
                ))

                if is_prose or (not is_code and len(stripped.split()) > 5):
                    # Likely an explanation, skip it
                    continue

            if not in_reasoning:
                cleaned_lines.append(line)

        code = '\n'.join(cleaned_lines)

        # Remove any leading/trailing whitespace
        code = code.strip()

        # Try to extract just the Python code if model mixed in other content
        # Look for the first import statement and take everything from there
        lines = code.split('\n')
        first_import_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                first_import_idx = i
                break

        if first_import_idx is not None and first_import_idx > 0:
            # There's content before the first import, remove it
            code = '\n'.join(lines[first_import_idx:])
            print(f"[ANALYSIS_AGENT] Removed {first_import_idx} lines of non-code before first import")

        # Validate the code can be parsed as Python
        try:
            compile(code, '<generated>', 'exec')
        except SyntaxError as e:
            # If there's a syntax error, log it but return the code anyway
            # The sandbox will catch it and report to user
            print(f"[ANALYSIS_AGENT] Warning: Generated code has syntax error at line {e.lineno}: {e.msg}")

        return code

    def _validate_code_safety(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that generated code is safe to execute.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check for forbidden patterns
        for pattern, description in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return (False, description)

        # Check imports are whitelisted
        import_pattern = r'(?:^|\n)\s*(?:import|from)\s+(\w+)'
        imports = re.findall(import_pattern, code)

        for imp in imports:
            # Extract base module name (e.g., 'pandas' from 'pandas.core')
            base_module = imp.split('.')[0]
            if base_module not in self.ALLOWED_IMPORTS:
                return (False, f"forbidden import: {imp}")

        return (True, None)
