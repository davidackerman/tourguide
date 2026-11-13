"""
Test script for analysis results API endpoints.

Tests the new /api/analysis/sessions and /api/analysis/session/{session_id} endpoints.
"""

import requests
import json
from pathlib import Path


def test_list_sessions():
    """Test listing analysis sessions."""
    print("\n" + "=" * 80)
    print("TEST: List Analysis Sessions")
    print("=" * 80)

    url = "http://localhost:8000/api/analysis/sessions"

    try:
        response = requests.get(url, params={"limit": 10})
        response.raise_for_status()

        data = response.json()
        print(f"\nStatus: {data.get('status')}")
        print(f"Total sessions: {data.get('total', 0)}")

        sessions = data.get('sessions', [])
        if sessions:
            print(f"\nRecent sessions (showing up to 10):\n")
            for i, session in enumerate(sessions, 1):
                print(f"{i}. Session: {session['session_id']}")
                print(f"   Created: {session['created_at']}")
                print(f"   Query: {session.get('query', 'N/A')[:80]}...")
                print(f"   Status: {session['status']}")
                print(f"   Plots: {session['plots_count']}")

                timing = session.get('timing', {})
                if timing:
                    print(f"   Timing:")
                    print(f"      Code gen: {timing.get('code_generation_seconds', 0):.3f}s")
                    print(f"      Execution: {timing.get('execution_seconds', 0):.3f}s")
                    print(f"      Total: {timing.get('total_seconds', 0):.3f}s")
                print()
        else:
            print("\nNo analysis sessions found.")

        return sessions

    except requests.RequestException as e:
        print(f"Error: {e}")
        return []


def test_get_session_details(session_id):
    """Test getting detailed session information."""
    print("\n" + "=" * 80)
    print(f"TEST: Get Session Details - {session_id}")
    print("=" * 80)

    url = f"http://localhost:8000/api/analysis/session/{session_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        if data.get('status') == 'ok':
            session = data.get('session', {})

            print(f"\nSession ID: {session.get('session_id')}")
            print(f"Created: {session.get('created_at')}")
            print(f"Status: {session.get('status')}")

            print(f"\nQuery:")
            print(f"  {session.get('query', 'N/A')}")

            print(f"\nGenerated Code:")
            print("-" * 80)
            code = session.get('code', '')
            if code:
                print(code)
            else:
                print("  (No code available)")
            print("-" * 80)

            print(f"\nExecution Results:")
            execution = session.get('execution', {})
            stdout = execution.get('stdout', '').strip()
            stderr = execution.get('stderr', '').strip()

            if stdout:
                print(f"  stdout:")
                for line in stdout.split('\n'):
                    print(f"    {line}")

            if stderr:
                print(f"  stderr:")
                for line in stderr.split('\n'):
                    print(f"    {line}")

            plots = execution.get('plots', [])
            if plots:
                print(f"\nGenerated Plots ({len(plots)}):")
                for plot in plots:
                    print(f"  - {plot}")
                    print(f"    URL: http://localhost:8000/api/analysis/plot/{session_id}/{plot}")

            timing = session.get('timing', {})
            if timing:
                print(f"\nTiming Breakdown:")
                print(f"  Code Generation: {timing.get('code_generation_seconds', 0):.3f}s")
                print(f"  Execution: {timing.get('execution_seconds', 0):.3f}s")
                print(f"  Total: {timing.get('total_seconds', 0):.3f}s")

        else:
            print(f"Error: {data.get('message')}")

    except requests.RequestException as e:
        print(f"Error: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS API TESTS")
    print("=" * 80)

    # Test listing sessions
    sessions = test_list_sessions()

    # Test getting details for the first session if available
    if sessions:
        test_get_session_details(sessions[0]['session_id'])

    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
