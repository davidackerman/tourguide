#!/usr/bin/env python3
"""
Tests for Analysis Mode components.

Tests the analysis agent and docker sandbox functionality
without requiring the full server to be running.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[SETUP] Loaded environment from {env_path}")

# Check if we need to start Ollama (for cluster nodes)
def ensure_ollama_if_needed():
    """Start Ollama if using local mode and it's not running."""
    use_local = os.getenv("USE_LOCAL", "false").lower() == "true"
    analysis_model = os.getenv("ANALYSIS_AI_MODEL", "")
    query_model = os.getenv("QUERY_AI_MODEL", "")

    has_google_key = bool(os.getenv("GOOGLE_API_KEY"))
    has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))

    # Check if we're using Ollama (any local model name in environment)
    ollama_models = ["qwen", "nemotron", "llama", "codellama", "deepseek", "mistral", "phi"]
    is_ollama_model = any(model in analysis_model.lower() for model in ollama_models) or \
                      any(model in query_model.lower() for model in ollama_models)

    # We need Ollama if:
    # 1. USE_LOCAL=true, OR
    # 2. An Ollama model is specified in ANALYSIS_AI_MODEL or QUERY_AI_MODEL, OR
    # 3. No API keys are configured (default to local)
    needs_ollama = use_local or is_ollama_model or (not has_google_key and not has_anthropic_key)

    if needs_ollama:
        print("[SETUP] Detected Ollama mode, ensuring Ollama is running...")
        print(f"[SETUP]   USE_LOCAL={use_local}")
        print(f"[SETUP]   ANALYSIS_AI_MODEL={analysis_model}")
        print(f"[SETUP]   Has API keys: Google={has_google_key}, Anthropic={has_anthropic_key}")

        try:
            from ollama_manager import ensure_ollama_running, preload_model

            if ensure_ollama_running():
                print("[SETUP] ✓ Ollama is available")
                # Preload the model to avoid cold-start issues
                model_name = analysis_model or query_model or "nemotron-3-nano"
                print(f"[SETUP] Preloading model: {model_name}")
                preload_model(model_name)
                return True
            else:
                print("[SETUP] ✗ Failed to start Ollama")
                return False
        except Exception as e:
            print(f"[SETUP] ✗ Error managing Ollama: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("[SETUP] Using API-based AI provider (Gemini or Claude)")
        print(f"[SETUP]   Has Google API key: {has_google_key}")
        print(f"[SETUP]   Has Anthropic API key: {has_anthropic_key}")
        return True

# Ensure Ollama before running tests
print()
ensure_ollama_if_needed()
print()

def test_analysis_agent():
    """Test the AnalysisAgent code generation."""
    print("\n" + "="*80)
    print("TEST 1: Analysis Agent - Code Generation")
    print("="*80)

    try:
        from analysis_agent import AnalysisAgent
        from organelle_db import OrganelleDatabase

        # Create a temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name

        try:
            # Initialize database (no CSV import for tests)
            db = OrganelleDatabase(db_path, csv_paths=[])

            # Add some test data directly using SQL
            conn = db._get_connection()
            cursor = conn.cursor()

            test_data = [
                ('1', 'mitochondria', 1000.0, 500.0, 100.0, 200.0, 300.0),
                ('2', 'nucleus', 5000.0, 2000.0, 150.0, 250.0, 350.0)
            ]

            cursor.executemany('''
                INSERT INTO organelles (object_id, organelle_type, volume, surface_area,
                                       position_x, position_y, position_z)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', test_data)
            conn.commit()
            conn.close()

            print(f"✓ Created test database with {len(test_data)} organelles")

            # Initialize analysis agent
            agent = AnalysisAgent(db=db, provider=None)

            if not agent.enabled:
                print("⚠ WARNING: Analysis agent not enabled (no AI provider configured)")
                print("  Set USE_LOCAL=true or provide GOOGLE_API_KEY/ANTHROPIC_API_KEY in .env")
                return False

            print(f"✓ Initialized AnalysisAgent (provider: {agent.provider})")

            # Test code generation
            test_query = "Show me a histogram of mitochondria volumes"
            print(f"\n  Query: '{test_query}'")

            result = agent.generate_analysis_code(test_query)

            if "error" in result:
                print(f"✗ FAILED: {result['error']}")
                return False

            code = result.get("code", "")

            if not code:
                print("✗ FAILED: No code generated")
                return False

            print(f"✓ Generated code ({len(code)} characters)")

            # Validate code contains expected elements
            required_elements = [
                "sqlite3",
                "pandas",
                "/data/organelles.db",
                "/output/"
            ]

            missing = [elem for elem in required_elements if elem not in code]
            if missing:
                print(f"✗ FAILED: Code missing required elements: {missing}")
                return False

            print("✓ Code contains required elements (sqlite3, pandas, /data/, /output/)")

            # Test safety validation
            unsafe_code = "import os\nos.system('rm -rf /')"
            is_safe, error_msg = agent._validate_code_safety(unsafe_code)

            if is_safe:
                print("✗ FAILED: Safety validation didn't catch unsafe code")
                return False

            print(f"✓ Safety validation working (blocked: {error_msg})")

            # Test safe code passes validation
            safe_code = "import pandas as pd\nimport sqlite3\nprint('hello')"
            is_safe, error_msg = agent._validate_code_safety(safe_code)

            if not is_safe:
                print(f"✗ FAILED: Safety validation rejected safe code: {error_msg}")
                return False

            print("✓ Safety validation passes safe code")

            print("\n✅ TEST 1 PASSED: AnalysisAgent working correctly")
            return True

        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)

    except Exception as e:
        print(f"\n✗ TEST 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_container_sandbox():
    """Test the ContainerSandbox execution environment."""
    print("\n" + "="*80)
    print("TEST 2: Container Sandbox - Execution Environment")
    print("="*80)

    try:
        from container_sandbox import ContainerSandbox

        # Create a temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name

        try:
            # Initialize database with test data
            from organelle_db import OrganelleDatabase
            db = OrganelleDatabase(db_path, csv_paths=[])

            # Add test organelle directly using SQL
            conn = db._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO organelles (object_id, organelle_type, volume, surface_area,
                                       position_x, position_y, position_z)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', ('1', 'mitochondria', 1500.0, 750.0, 100.0, 200.0, 300.0))
            conn.commit()
            conn.close()

            print(f"✓ Created test database at {db_path}")

            # Initialize container sandbox
            sandbox = ContainerSandbox(db_path=db_path, timeout=30)
            print("✓ Initialized ContainerSandbox")

            # Test simple code execution
            simple_code = """
import sqlite3
import pandas as pd

conn = sqlite3.connect('/data/organelles.db')
df = pd.read_sql_query("SELECT * FROM organelles", conn)
conn.close()

print(f"Found {len(df)} organelles")
print(f"Total volume: {df['volume'].sum():.2f} nm³")
"""

            print("\n  Executing simple database query...")
            session_id = "test_session_1"

            result = sandbox.execute(code=simple_code, session_id=session_id)

            if result["status"] != "success":
                print(f"✗ FAILED: Execution failed")
                print(f"  Status: {result['status']}")
                print(f"  Stderr: {result['stderr']}")
                return False

            print(f"✓ Execution successful ({result['execution_time']:.2f}s)")

            if "Found 1 organelles" not in result["stdout"]:
                print(f"✗ FAILED: Unexpected output")
                print(f"  Output: {result['stdout']}")
                return False

            print(f"✓ Output correct: {result['stdout'].strip()}")

            # Test plot generation
            plot_code = """
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('/data/organelles.db')
df = pd.read_sql_query("SELECT volume FROM organelles", conn)
conn.close()

plt.figure(figsize=(8, 6))
plt.hist(df['volume'], bins=10)
plt.xlabel('Volume (nm³)')
plt.ylabel('Count')
plt.title('Volume Distribution')
plt.savefig('/output/test_plot.png', dpi=100, bbox_inches='tight')

print("Plot saved successfully")
"""

            print("\n  Executing plot generation code...")
            session_id = "test_session_2"

            result = sandbox.execute(code=plot_code, session_id=session_id)

            if result["status"] != "success":
                print(f"✗ FAILED: Plot generation failed")
                print(f"  Stderr: {result['stderr']}")
                return False

            print(f"✓ Plot generation successful")

            if "test_plot.png" not in result["plots"]:
                print(f"✗ FAILED: Plot file not found")
                print(f"  Plots: {result['plots']}")
                return False

            print(f"✓ Plot file created: {result['plots']}")

            # Verify plot file exists
            plot_path = Path(result["output_path"]) / "test_plot.png"
            if not plot_path.exists():
                print(f"✗ FAILED: Plot file doesn't exist at {plot_path}")
                return False

            print(f"✓ Plot file verified at {plot_path}")

            # Test timeout
            timeout_code = """
import time
time.sleep(100)  # Should timeout
"""

            print("\n  Testing timeout mechanism...")
            session_id = "test_session_3"

            result = sandbox.execute(code=timeout_code, session_id=session_id)

            if result["status"] != "timeout":
                print(f"✗ FAILED: Timeout not detected")
                print(f"  Status: {result['status']}")
                return False

            print(f"✓ Timeout mechanism working ({result['execution_time']:.2f}s)")

            print("\n✅ TEST 2 PASSED: ContainerSandbox working correctly")
            return True

        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)

    except Exception as e:
        print(f"\n✗ TEST 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete analysis pipeline: query → code generation → execution."""
    print("\n" + "="*80)
    print("TEST 3: Full Pipeline - Query to Results")
    print("="*80)

    try:
        import time as time_module
        from analysis_agent import AnalysisAgent
        from container_sandbox import ContainerSandbox
        from organelle_db import OrganelleDatabase

        # Use the existing real database instead of creating test data
        db_path = Path(__file__).parent.parent / "organelle_data" / "organelles.db"

        if not db_path.exists():
            print(f"✗ SKIPPED: Real database not found at {db_path}")
            print("  Please ensure organelle_data/organelles.db exists")
            return None

        try:
            # Track total timing
            pipeline_start = time_module.time()

            # Open the existing database
            db = OrganelleDatabase(str(db_path), csv_paths=[])

            # Check what data we have
            conn = db._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as total, organelle_type FROM organelles GROUP BY organelle_type ORDER BY total DESC LIMIT 5')
            counts = cursor.fetchall()
            cursor.execute('SELECT COUNT(*) FROM organelles')
            total_count = cursor.fetchone()[0]
            conn.close()

            print(f"✓ Using real database with {total_count} organelles:")
            for count, org_type in counts:
                print(f"    {org_type}: {count}")

            # Initialize components
            agent = AnalysisAgent(db=db, provider=None)
            if not agent.enabled:
                print("⚠ WARNING: Analysis agent not enabled - skipping full pipeline test")
                return None  # Skip test

            sandbox = ContainerSandbox(db_path=str(db_path), timeout=30)
            print(f"✓ Initialized components (agent provider: {agent.provider})")

            # Test query
            query = "Create a scatter plot of volume vs surface area for mitochondria and show statistics"
            print(f"\n  Query: '{query}'")

            # Generate code with timing
            print("\n  [TIMING] Code Generation...")
            code_gen_start = time_module.time()
            code_result = agent.generate_analysis_code(query)
            code_gen_time = time_module.time() - code_gen_start

            if "error" in code_result:
                print(f"✗ FAILED: Code generation error: {code_result['error']}")
                return False

            print(f"✓ Generated code ({len(code_result['code'])} chars) in {code_gen_time:.3f}s")

            # Execute code
            print("\n  [TIMING] Code Execution...")
            session_id = "test_pipeline_real"
            exec_result = sandbox.execute(
                code=code_result["code"],
                session_id=session_id
            )

            if exec_result["status"] != "success":
                print(f"✗ FAILED: Execution failed")
                print(f"  Status: {exec_result['status']}")
                print(f"  Stderr: {exec_result['stderr']}")
                print(f"\nGenerated code:")
                print(code_result['code'])
                return False

            exec_time = exec_result['execution_time']
            print(f"✓ Execution successful in {exec_time:.3f}s")

            # Check results
            if exec_result["plots"]:
                print(f"✓ Generated {len(exec_result['plots'])} plot(s): {exec_result['plots']}")
            else:
                print("  Note: No plots generated (may be statistics-only query)")

            if exec_result["stdout"]:
                # Show first few lines of output
                stdout_lines = exec_result['stdout'].strip().split('\n')
                preview = '\n'.join(stdout_lines[:10])
                if len(stdout_lines) > 10:
                    preview += f"\n  ... ({len(stdout_lines) - 10} more lines)"
                print(f"✓ Statistics output:\n{preview}")
            else:
                print("  Note: No statistics output")

            print(f"✓ Results saved to: {exec_result['output_path']}")

            # Calculate and display timing breakdown
            pipeline_total = time_module.time() - pipeline_start

            print("\n  [TIMING SUMMARY]")
            print(f"    Code Generation:  {code_gen_time:6.3f}s ({code_gen_time/pipeline_total*100:5.1f}%)")
            print(f"    Code Execution:   {exec_time:6.3f}s ({exec_time/pipeline_total*100:5.1f}%)")
            print(f"    Other (overhead): {pipeline_total - code_gen_time - exec_time:6.3f}s ({(pipeline_total - code_gen_time - exec_time)/pipeline_total*100:5.1f}%)")
            print(f"    " + "-" * 40)
            print(f"    TOTAL:            {pipeline_total:6.3f}s (100.0%)")

            print("\n✅ TEST 3 PASSED: Full pipeline working correctly with real data")
            return True

        finally:
            # Don't delete the real database!
            pass

    except Exception as e:
        print(f"\n✗ TEST 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ANALYSIS MODE TEST SUITE")
    print("="*80)

    # Check environment
    print("\nEnvironment Check:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working directory: {os.getcwd()}")

    # Check for AI configuration
    use_local = os.getenv("USE_LOCAL", "false").lower() == "true"
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if use_local:
        print("  AI Provider: Ollama (local)")
    elif has_google:
        print("  AI Provider: Google Gemini")
    elif has_anthropic:
        print("  AI Provider: Anthropic Claude")
    else:
        print("  AI Provider: ⚠ NONE CONFIGURED")
        print("    Set USE_LOCAL=true or provide API key in .env")

    # Check Container Runtime
    import subprocess
    has_docker = False
    has_apptainer = False

    try:
        result = subprocess.run(["docker", "version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            has_docker = True
            print("  Docker: ✓ Available")
    except Exception:
        pass

    try:
        result = subprocess.run(["apptainer", "version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            has_apptainer = True
            print("  Apptainer: ✓ Available")
    except Exception:
        pass

    if not has_docker and not has_apptainer:
        print("  Container Runtime: ✗ None available")
        print("    Install Docker: sudo apt-get install docker.io")
        print("    OR install Apptainer: pixi install")

    # Run tests
    results = []

    # Test 1: Analysis Agent
    result = test_analysis_agent()
    results.append(("Analysis Agent", result))

    # Test 2: Container Sandbox
    result = test_container_sandbox()
    results.append(("Container Sandbox", result))

    # Test 3: Full Pipeline
    result = test_full_pipeline()
    if result is not None:  # None means skipped
        results.append(("Full Pipeline", result))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, result in results:
        if result is True:
            print(f"✅ {name}: PASSED")
        elif result is False:
            print(f"✗ {name}: FAILED")
        else:
            print(f"⊘ {name}: SKIPPED")

    all_passed = all(r in [True, None] for _, r in results)
    any_failed = any(r is False for _, r in results)

    if all_passed and not any_failed:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    elif any_failed:
        print("\n❌ SOME TESTS FAILED")
        return 1
    else:
        print("\n⚠ TESTS INCOMPLETE (check configuration)")
        return 2


if __name__ == "__main__":
    sys.exit(main())
