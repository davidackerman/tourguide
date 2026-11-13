"""
Analysis Results Management Module.

Handles listing, tracking, and displaying analysis session results with metadata,
timing information, and generated artifacts.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class AnalysisResultsManager:
    """Manages analysis results sessions and metadata."""

    def __init__(self, results_dir: Path = None):
        """
        Initialize results manager.

        Args:
            results_dir: Base directory for analysis results (default: ./analysis_results)
        """
        if results_dir is None:
            results_dir = Path("./analysis_results")
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)

    def save_session_metadata(
        self,
        session_id: str,
        query: str,
        code: str,
        execution_result: Dict[str, Any],
        code_generation_time: float = 0.0,
    ):
        """
        Save metadata for an analysis session.

        Args:
            session_id: Unique session identifier
            query: User's natural language query
            code: Generated Python code
            execution_result: Result dictionary from container execution
            code_generation_time: Time taken to generate code (seconds)
        """
        session_dir = self.results_dir / session_id
        session_dir.mkdir(exist_ok=True)

        metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "query": query,
            "code": code,
            "status": execution_result.get("status", "unknown"),
            "timing": {
                "code_generation_seconds": round(code_generation_time, 3),
                "execution_seconds": round(execution_result.get("execution_time", 0), 3),
                "total_seconds": round(code_generation_time + execution_result.get("execution_time", 0), 3),
            },
            "execution": {
                "stdout": execution_result.get("stdout", ""),
                "stderr": execution_result.get("stderr", ""),
                "plots": execution_result.get("plots", []),
            },
        }

        metadata_file = session_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[ANALYSIS_RESULTS] Metadata saved for {session_id}")

    def list_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all analysis sessions sorted by creation time (newest first).

        Args:
            limit: Maximum number of sessions to return (None = all)

        Returns:
            List of session metadata dictionaries with summary information
        """
        sessions = []

        for session_dir in self.results_dir.iterdir():
            if not session_dir.is_dir():
                continue

            metadata_file = session_dir / "metadata.json"

            # Load metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    # Create summary
                    summary = {
                        "session_id": metadata.get("session_id", session_dir.name),
                        "created_at": metadata.get("created_at"),
                        "query": metadata.get("query", "")[:100],  # Truncate long queries
                        "status": metadata.get("status", "unknown"),
                        "plots_count": len(metadata.get("execution", {}).get("plots", [])),
                        "timing": metadata.get("timing", {}),
                    }
                    sessions.append(summary)
                except Exception as e:
                    print(f"[ANALYSIS_RESULTS] Error reading {session_dir.name}: {e}")
                    continue
            else:
                # Fallback: create basic info from directory
                try:
                    stat = session_dir.stat()
                    plots = list(session_dir.glob("*.png")) + list(session_dir.glob("*.html"))

                    summary = {
                        "session_id": session_dir.name,
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "query": "",
                        "status": "completed" if plots else "unknown",
                        "plots_count": len(plots),
                        "timing": {},
                    }
                    sessions.append(summary)
                except Exception as e:
                    print(f"[ANALYSIS_RESULTS] Error reading {session_dir.name}: {e}")
                    continue

        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x["created_at"], reverse=True)

        if limit:
            sessions = sessions[:limit]

        return sessions

    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Full metadata dictionary or None if not found
        """
        session_dir = self.results_dir / session_id

        if not session_dir.exists():
            return None

        metadata_file = session_dir / "metadata.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ANALYSIS_RESULTS] Error reading metadata for {session_id}: {e}")
                return None

        # Fallback: construct from directory contents
        try:
            plots = [p.name for p in session_dir.glob("*.png")] + [p.name for p in session_dir.glob("*.html")]
            code_file = session_dir / "analysis.py"
            stdout_file = session_dir / "stdout.log"
            stderr_file = session_dir / "stderr.log"

            code = code_file.read_text() if code_file.exists() else ""
            stdout = stdout_file.read_text() if stdout_file.exists() else ""
            stderr = stderr_file.read_text() if stderr_file.exists() else ""

            return {
                "session_id": session_id,
                "created_at": datetime.fromtimestamp(session_dir.stat().st_mtime).isoformat(),
                "query": "",
                "code": code,
                "status": "completed" if plots else "error",
                "timing": {},
                "execution": {
                    "stdout": stdout,
                    "stderr": stderr,
                    "plots": plots,
                },
            }
        except Exception as e:
            print(f"[ANALYSIS_RESULTS] Error constructing fallback metadata for {session_id}: {e}")
            return None
