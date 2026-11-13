"""
Migrate existing analysis results to include metadata.json files.
"""

import sys
sys.path.insert(0, 'server')

from analysis_results import AnalysisResultsManager
import json
from pathlib import Path
from datetime import datetime


def main():
    """Add metadata to existing analysis sessions."""
    manager = AnalysisResultsManager()

    results_dir = Path('analysis_results')
    if not results_dir.exists():
        print("No analysis_results directory found")
        return

    migrated_count = 0

    for session_dir in results_dir.iterdir():
        if not session_dir.is_dir():
            continue

        metadata_file = session_dir / 'metadata.json'
        if metadata_file.exists():
            print(f"Skipping {session_dir.name} (already has metadata)")
            continue

        # Create basic metadata for existing sessions
        code_file = session_dir / 'analysis.py'
        stdout_file = session_dir / 'stdout.log'
        stderr_file = session_dir / 'stderr.log'

        plots = [p.name for p in session_dir.glob('*.png')] + [p.name for p in session_dir.glob('*.html')]

        metadata = {
            'session_id': session_dir.name,
            'created_at': datetime.fromtimestamp(session_dir.stat().st_mtime).isoformat(),
            'query': '(migrated from old session)',
            'code': code_file.read_text() if code_file.exists() else '',
            'status': 'completed' if plots else 'unknown',
            'timing': {},
            'execution': {
                'stdout': stdout_file.read_text() if stdout_file.exists() else '',
                'stderr': stderr_file.read_text() if stderr_file.exists() else '',
                'plots': plots
            }
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Created metadata for {session_dir.name}")
        migrated_count += 1

    print(f"\nMigration complete: {migrated_count} sessions updated")


if __name__ == "__main__":
    main()
