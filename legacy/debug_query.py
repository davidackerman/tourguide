#!/usr/bin/env python3
"""
Debug script for testing the query agent from the command line.
This allows you to test queries without going through the web interface.

Usage:
    python debug_query.py "What is the biggest mitochondria?"
    python debug_query.py  # Interactive mode
"""

import sys
import os
from pathlib import Path
import time

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from organelle_db import OrganelleDatabase
from query_agent import QueryAgent
from dotenv import load_dotenv
import json

def print_separator(char="-", length=80):
    print(char * length)

def pretty_print_results(results):
    """Pretty print database results."""
    if not results:
        print("  (No results)")
        return

    if len(results) == 1:
        print("  Result:")
        for key, value in results[0].items():
            print(f"    {key}: {value}")
    else:
        print(f"  {len(results)} results:")
        for i, row in enumerate(results, 1):
            print(f"  Row {i}:")
            for key, value in row.items():
                print(f"    {key}: {value}")
            if i < len(results):
                print()

def run_query(agent: QueryAgent, query: str):
    """Run a single query and display all intermediate steps with timing."""
    print_separator("=")
    print(f"QUERY: {query}")
    print_separator("=")
    print()

    # Process the query with timing
    print("[1] Processing query...")
    start_time = time.time()
    result = agent.process_query(query)
    total_time = time.time() - start_time
    print()

    # Display the generated SQL
    print_separator()
    print("GENERATED SQL:")
    print_separator()
    print(result.get("sql", "(No SQL generated)"))
    print()

    # Display raw database results
    print_separator()
    print("DATABASE RESULTS:")
    print_separator()
    results = result.get("results", [])
    pretty_print_results(results)
    print()

    # Display query type
    print_separator()
    print("QUERY TYPE:")
    print_separator()
    print(f"  {result.get('type', 'unknown')}")
    print()

    # Display navigation info if present
    if "navigation" in result:
        print_separator()
        print("NAVIGATION COMMAND (would update Neuroglancer):")
        print_separator()
        nav = result["navigation"]
        print(f"  Position: {nav.get('position')}")
        print(f"  Scale: {nav.get('scale')}")
        print(f"  Object ID: {nav.get('object_id')}")
        print()
        print("  Python code that would execute:")
        print(f"    with viewer.txn() as s:")
        print(f"        s.position = {nav.get('position')}")
        print(f"        s.crossSectionScale = {nav.get('scale')}")
        print()

    # Display visualization info if present
    if "visualization" in result:
        print_separator()
        print("VISUALIZATION COMMAND (would update Neuroglancer):")
        print_separator()

        viz = result["visualization"]

        # Handle both single command (dict) and multiple commands (list)
        viz_commands = viz if isinstance(viz, list) else [viz]

        print(f"  Number of layer commands: {len(viz_commands)}")
        print()

        for i, viz_cmd in enumerate(viz_commands, 1):
            layer_name = viz_cmd.get('layer_name')
            segment_ids = viz_cmd.get('segment_ids', [])
            action = viz_cmd.get('action', 'show_only')

            if len(viz_commands) > 1:
                print(f"  Command {i}:")
                prefix = "    "
            else:
                prefix = "  "

            print(f"{prefix}Layer: {layer_name}")
            print(f"{prefix}Action: {action}")
            print(f"{prefix}Segment IDs: {segment_ids[:5]}{'...' if len(segment_ids) > 5 else ''}")
            print(f"{prefix}Total segments: {len(segment_ids)}")
            if i < len(viz_commands):
                print()

        print()
        print("  Python code that would execute:")
        print(f"    with viewer.txn() as s:")
        for viz_cmd in viz_commands:
            layer_name = viz_cmd.get('layer_name')
            segment_ids = viz_cmd.get('segment_ids', [])
            action = viz_cmd.get('action', 'show_only')

            if action == "show_only":
                print(f"        s.layers['{layer_name}'].segments = {{{', '.join(repr(s) for s in segment_ids[:3])}{', ...' if len(segment_ids) > 3 else ''}}}")
            elif action == "add":
                print(f"        current = set(s.layers['{layer_name}'].segments)")
                print(f"        s.layers['{layer_name}'].segments = current | {{{', '.join(repr(s) for s in segment_ids[:3])}}}")
            elif action == "remove":
                print(f"        current = set(s.layers['{layer_name}'].segments)")
                print(f"        s.layers['{layer_name}'].segments = current - {{{', '.join(repr(s) for s in segment_ids[:3])}}}")
        print()

    # Display AI interactions if available
    if "ai_interactions" in result and result["ai_interactions"]:
        print_separator()
        print("AI INTERACTIONS:")
        print_separator()
        print(f"  Model: {result.get('model', 'unknown')}")
        print(f"  Total interactions: {len(result['ai_interactions'])}")
        print()

        for i, interaction in enumerate(result['ai_interactions'], 1):
            interaction_type = interaction.get('type', 'unknown')
            print(f"  [{i}] {interaction_type.upper().replace('_', ' ')}")
            print()

            # Show prompt (truncated)
            prompt = interaction.get('prompt', '')
            print("    PROMPT:")
            if len(prompt) > 500:
                print(f"      {prompt[:500]}...")
                print(f"      (truncated, {len(prompt)} chars total)")
            else:
                for line in prompt.split('\n'):
                    print(f"      {line}")
            print()

            # Show response
            response = interaction.get('response', '')
            print("    RESPONSE:")
            for line in response.split('\n'):
                print(f"      {line}")
            print()

            # Show cleaned SQL if available
            if 'cleaned_sql' in interaction:
                print("    CLEANED SQL:")
                print(f"      {interaction['cleaned_sql']}")
                print()

            # Show retry info if available
            if interaction.get('retry'):
                print(f"    ⚠️  RETRY ATTEMPT (error: {interaction.get('previous_error', 'unknown')})")
                print()

            if i < len(result['ai_interactions']):
                print_separator("-")
                print()

    # Display final answer
    print_separator()
    print("FINAL ANSWER:")
    print_separator()
    print(f"  {result.get('answer', '(No answer generated)')}")
    print()

    # Display timing breakdown if available
    print_separator()
    print("PERFORMANCE:")
    print_separator()
    if "timing" in result:
        timing = result["timing"]
        print(f"  SQL Generation:     {timing.get('sql_generation', 0):.3f}s")
        print(f"  Query Execution:    {timing.get('query_execution', 0):.3f}s")
        print(f"  Answer Formatting:  {timing.get('answer_formatting', 0):.3f}s")
        print(f"  Total:              {timing.get('total', total_time):.3f}s")
    else:
        print(f"  Total Time:         {total_time:.3f}s")
    print()

    print_separator("=")
    print()

def interactive_mode(agent: QueryAgent):
    """Run in interactive mode, prompting for queries."""
    print("=" * 80)
    print("Query Agent Debug Mode (Interactive)")
    print("=" * 80)
    print("Enter your queries below. Type 'quit', 'exit', or press Ctrl+C to stop.")
    print()

    while True:
        try:
            query = input("\nQuery> ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("\nExiting...")
                break

            print()
            run_query(agent, query)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            print()

def main():
    # Load environment variables
    load_dotenv()

    # Get configuration
    csv_paths_str = os.getenv("ORGANELLE_CSV_PATHS", "")
    if not csv_paths_str:
        print("ERROR: ORGANELLE_CSV_PATHS not set in .env file")
        print("Please set the paths to your organelle CSV files.")
        sys.exit(1)

    csv_paths = [p.strip() for p in csv_paths_str.split(",") if p.strip()]
    db_path = os.getenv("ORGANELLE_DB_PATH", "organelles.db")
    model = os.getenv("QUERY_AI_MODEL", "nemotron-3-nano")

    # Check for --verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if verbose:
        sys.argv = [arg for arg in sys.argv if arg not in ["--verbose", "-v"]]

    print("Initializing database...")
    print(f"  Database path: {db_path}")
    print(f"  CSV files: {len(csv_paths)}")
    for path in csv_paths:
        print(f"    - {path}")
    print()

    # Initialize database
    db = OrganelleDatabase(db_path, csv_paths)

    print(f"Database initialized successfully!")
    print(f"  Total organelles: {db.get_row_count()}")
    print(f"  Available types: {', '.join(db.get_available_organelle_types())}")
    print()

    print(f"Initializing query agent with model: {model}")
    if verbose:
        print(f"  Verbose mode: ENABLED")
    agent = QueryAgent(db, model=model, verbose=verbose)
    print()

    # Check if query provided as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(agent, query)
    else:
        interactive_mode(agent)

if __name__ == "__main__":
    main()
