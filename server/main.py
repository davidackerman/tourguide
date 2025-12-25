"""
Main entry point for Neuroglancer Live Stream server.
Starts Neuroglancer viewer, screenshot loop, and FastAPI streaming server.
"""

import asyncio
import argparse
import sys
import socket
import os
from dotenv import load_dotenv
from ng import NG_StateTracker
from stream import create_app
from organelle_db import OrganelleDatabase
from query_agent import QueryAgent
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Set OLLAMA_MODELS path if specified in .env
ollama_models_path = os.getenv("OLLAMA_MODELS")
if ollama_models_path:
    os.environ["OLLAMA_MODELS"] = ollama_models_path
    print(f"[OLLAMA] Using custom models path: {ollama_models_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Neuroglancer Live Stream Server")
    parser.add_argument(
        "--ng-host",
        default="0.0.0.0",
        help="Neuroglancer bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--ng-port", type=int, default=9999, help="Neuroglancer port (default: 9999)"
    )
    parser.add_argument(
        "--web-host",
        default="0.0.0.0",
        help="Web server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--web-port", type=int, default=8090, help="Web server port (default: 8090)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.1,
        help="Maximum screenshot frame rate (default: 0.1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (uses example narration instead of AI)",
    )
    args = parser.parse_args()

    # Set debug environment variable if flag is provided
    if args.debug:
        os.environ["DEBUG_MODE"] = "true"
        print("[DEBUG] Debug mode enabled - using example narrations", flush=True)

    print("=" * 60, flush=True)
    print("Neuroglancer Live Stream Server", flush=True)
    print("=" * 60, flush=True)

    # Initialize Neuroglancer tracker
    print(
        f"\n[1/4] Starting Neuroglancer viewer on {args.ng_host}:{args.ng_port}...",
        flush=True,
    )
    tracker = NG_StateTracker(bind_address=args.ng_host, port=args.ng_port)
    print(f"      Neuroglancer URL: {tracker.get_url()}", flush=True)

    # Initialize organelle database (NEW)
    print(f"\n[2/4] Initializing organelle database...", flush=True)
    csv_paths_str = os.getenv("ORGANELLE_CSV_PATHS", "")
    csv_paths = [p.strip() for p in csv_paths_str.split(",") if p.strip()]

    query_agent = None
    if csv_paths:
        try:
            db_path = os.getenv("ORGANELLE_DB_PATH", "./organelle_data/organelles.db")
            db = OrganelleDatabase(db_path=db_path, csv_paths=csv_paths)

            row_count = db.get_row_count()
            organelle_types = db.get_available_organelle_types()

            print(f"      Database: {db_path}", flush=True)
            print(f"      Total organelles: {row_count}", flush=True)
            print(f"      Organelle types: {', '.join(organelle_types)}", flush=True)

            query_model = os.getenv("QUERY_AI_MODEL", "nemotron")
            query_agent = QueryAgent(db=db, model=query_model)
            print(f"      Query mode: ENABLED (model: {query_model})", flush=True)
        except Exception as e:
            print(f"      Failed to initialize database: {e}", flush=True)
            print(f"      Query mode: DISABLED", flush=True)
    else:
        print(f"      No ORGANELLE_CSV_PATHS configured in .env", flush=True)
        print(f"      Query mode: DISABLED", flush=True)

    # Use client-side screenshot capture (no cross-origin data)
    print(f"\n[3/4] Using client-side screenshot capture", flush=True)
    print(f"      No external data loaded (avoids WebGL canvas tainting)", flush=True)
    # tracker.start_screenshot_loop(max_fps=0.033)  # Disabled - using client-side

    # Get hostname for display
    hostname = socket.gethostname()

    # Create and run FastAPI app
    print(
        f"\n[4/4] Starting web server on {args.web_host}:{args.web_port}...", flush=True
    )
    app = create_app(tracker, query_agent=query_agent)
    print(f"      Web panel: http://{hostname}:{args.web_port}/", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Server ready!", flush=True)
    print("=" * 60, flush=True)
    print(f"\nOpen web panel at:", flush=True)
    print(f"  - http://{hostname}:{args.web_port}/", flush=True)
    print(f"  - http://localhost:{args.web_port}/ (local only)", flush=True)
    print(f"\nNeuroglancer URL (embedded in web panel):", flush=True)
    print(f"  - {tracker.get_url()}", flush=True)
    print("\nPress Ctrl+C to stop\n", flush=True)

    try:
        uvicorn.run(app, host=args.web_host, port=args.web_port, log_level="info")
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        tracker.stop_screenshot_loop()
        print("Done.")


if __name__ == "__main__":
    main()
