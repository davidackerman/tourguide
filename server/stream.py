"""
FastAPI server with WebSocket streaming.
Stage 3: WebSocket endpoint that streams frames + state to browser.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
import asyncio
import json
import time
from pathlib import Path
from typing import Set
import httpx
import base64
from datetime import datetime

from recording import RecordingManager, MovieCompiler
from analysis_results import AnalysisResultsManager

# Will be set by main.py
ng_tracker = None


def create_app(tracker, query_agent=None) -> FastAPI:
    """Create FastAPI app with WebSocket streaming."""
    global ng_tracker
    ng_tracker = tracker

    app = FastAPI(title="Neuroglancer Live Stream")

    # Store query agent reference
    app.state.query_agent = query_agent

    # Initialize analysis components
    analysis_agent = None
    container_sandbox = None
    analysis_results_manager = None

    if query_agent:  # Only if database configured
        try:
            from analysis_agent import AnalysisAgent
            from container_sandbox import ContainerSandbox
            import os

            analysis_agent = AnalysisAgent(
                db=query_agent.db,
                provider=None  # Auto-detect like narrator
            )
            container_sandbox = ContainerSandbox(
                db_path=os.getenv("ORGANELLE_DB_PATH", "./organelle_data/organelles.db"),
                timeout=int(os.getenv("CONTAINER_TIMEOUT", os.getenv("DOCKER_TIMEOUT", "60")))
            )
            analysis_results_manager = AnalysisResultsManager()
            print("[ANALYSIS] Analysis mode initialized", flush=True)
        except Exception as e:
            print(f"[ANALYSIS] Failed to initialize analysis mode: {e}", flush=True)
            import traceback
            traceback.print_exc()

    app.state.analysis_agent = analysis_agent
    app.state.container_sandbox = container_sandbox
    app.state.analysis_results_manager = analysis_results_manager

    # Track active WebSocket connections
    active_connections: Set[WebSocket] = set()

    # Queue for frame updates
    frame_queue = asyncio.Queue()

    # Queue for narration updates (Stage 4)
    narration_queue = asyncio.Queue()

    # Recording manager for movie creation
    recording_manager = RecordingManager()

    # Mode state (explore vs query)
    # Default to query mode if query_agent is available, otherwise explore
    default_mode = "query" if query_agent else "explore"
    current_mode = {"mode": default_mode}

    # Narration is controlled by client's voice toggle (generate_audio flag in requests)
    # Server-side narration_enabled kept at True for backward compatibility
    ng_tracker.narration_enabled = True
    print(f"[INIT] Starting in {default_mode} mode (audio generation controlled by client)", flush=True)

    async def broadcast_narration(narration_text: str):
        """Broadcast narration to all connected clients."""
        message = {
            "type": "narration",
            "text": narration_text,
            "timestamp": time.time(),
        }
        disconnected = set()
        for connection in active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"[WS] Failed to send narration to client: {e}")
                disconnected.add(connection)
        # Clean up disconnected clients
        active_connections.difference_update(disconnected)

    def narration_callback(narration_text: str):
        """Callback for narrator to queue narration messages."""
        # Queue the narration for async broadcasting
        try:
            narration_queue.put_nowait(narration_text)
        except asyncio.QueueFull:
            print("[WARN] Narration queue full, dropping message")

    # Set the narrator callback on the tracker
    ng_tracker.narrator_callback = narration_callback

    # Background task to broadcast narrations
    async def narration_broadcaster():
        """Background task that broadcasts queued narrations."""
        while True:
            try:
                narration_text = await narration_queue.get()
                await broadcast_narration(narration_text)
                print(f"[NARRATION] Broadcasted: {narration_text[:100]}...")
            except Exception as e:
                print(f"[ERROR] Narration broadcaster error: {e}")
                await asyncio.sleep(0.5)

    @app.on_event("startup")
    async def startup_event():
        """Start background tasks."""
        asyncio.create_task(narration_broadcaster())

    @app.get("/")
    async def serve_index():
        """Serve the main web page."""
        web_dir = Path(__file__).parent.parent / "web"
        response = FileResponse(web_dir / "index.html")
        # Prevent caching to ensure browser always gets latest version
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/api/ng-url")
    async def get_ng_url():
        """Get the proxied Neuroglancer viewer URL for iframe embedding."""
        # Return PROXIED URL through same origin (port 8090)
        ng_url = ng_tracker.get_url()
        # Extract the path portion (everything after the host)
        if "/v/" in ng_url:
            path = ng_url.split("/v/", 1)[1]
            proxied_url = f"/ng-proxy/v/{path}"
        else:
            proxied_url = "/ng-proxy/"
        return {"url": proxied_url}

    @app.api_route(
        "/ng-proxy/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    )
    async def neuroglancer_proxy(path: str, request: Request):
        """Proxy all requests to Neuroglancer server to avoid cross-origin issues."""
        # Get the Neuroglancer base URL (http://host:port)
        ng_url = ng_tracker.get_url()
        # Extract just protocol + host + port
        if "://" in ng_url:
            parts = ng_url.split("://")
            protocol = parts[0]
            rest = parts[1]
            if "/" in rest:
                host_port = rest.split("/", 1)[0]
            else:
                host_port = rest
            base_url = f"{protocol}://{host_port}"
        else:
            # Fallback
            base_url = "http://localhost:9999"

        # Forward the request to Neuroglancer
        proxy_url = f"{base_url}/{path}"

        # Get query parameters
        query_string = str(request.url.query)
        if query_string:
            proxy_url += f"?{query_string}"

        # Special handling for Server-Sent Events (SSE) /events endpoint
        if path.startswith("events/"):
            # For SSE, we need to stream the response
            async def event_stream():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        method=request.method,
                        url=proxy_url,
                        headers={
                            k: v
                            for k, v in request.headers.items()
                            if k.lower() not in ["host", "connection"]
                        },
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # For other requests, use normal proxying
        try:
            async with httpx.AsyncClient() as client:
                # Forward request
                response = await client.request(
                    method=request.method,
                    url=proxy_url,
                    headers={
                        k: v
                        for k, v in request.headers.items()
                        if k.lower() not in ["host", "connection"]
                    },
                    content=(
                        await request.body()
                        if request.method in ["POST", "PUT", "PATCH"]
                        else None
                    ),
                    timeout=30.0,
                )

                # Inject screenshot handler script into HTML responses
                content = response.content
                content_type = response.headers.get("content-type", "")
                headers = dict(response.headers)

                if "text/html" in content_type:
                    try:
                        html = content.decode("utf-8")

                        # Inject WebGL monkey-patch FIRST (before any other scripts)
                        webgl_patch = """<script>
(function() {
    console.log('[WEBGL-PATCH] Monkey-patching getContext to force preserveDrawingBuffer');
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, attributes) {
        if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') {
            attributes = attributes || {};
            attributes.preserveDrawingBuffer = true;
            console.log('[WEBGL-PATCH] Forcing preserveDrawingBuffer=true for', type);
        }
        return originalGetContext.call(this, type, attributes);
    };
})();
</script>"""

                        # Inject at the very beginning of <head> or <html>
                        if "<head>" in html:
                            html = html.replace("<head>", f"<head>{webgl_patch}")
                        elif "<html>" in html:
                            html = html.replace("<html>", f"<html>{webgl_patch}")
                        else:
                            html = webgl_patch + html

                        # Also inject the screenshot handler
                        script_content = open(
                            Path(__file__).parent.parent
                            / "web"
                            / "ng-screenshot-handler.js",
                            "r",
                        ).read()
                        script_tag = f"<script>{script_content}</script>"
                        if "</head>" in html:
                            html = html.replace("</head>", f"{script_tag}</head>")
                        elif "</body>" in html:
                            html = html.replace("</body>", f"{script_tag}</body>")
                        else:
                            html += script_tag
                        content = html.encode("utf-8")
                        # Update Content-Length header
                        headers["content-length"] = str(len(content))
                        print(
                            f"[PROXY] Injected WebGL patch and screenshot handler into HTML"
                        )
                    except Exception as e:
                        print(f"[PROXY] Failed to inject script: {e}")

                # Return response
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.headers.get("content-type"),
                )
        except Exception as e:
            print(f"[PROXY ERROR] Failed to proxy {proxy_url}: {e}")
            return Response(content=f"Proxy error: {str(e)}", status_code=500)

    @app.post("/api/screenshot")
    async def receive_screenshot(request: Request):
        """Receive screenshot from browser client."""
        try:
            data = await request.json()
            jpeg_b64 = data.get("jpeg_b64")
            timestamp = data.get("timestamp", time.time())
            generate_audio = data.get("generate_audio", True)  # Default to True for backward compatibility
            manual_capture = data.get("manual_capture", False)  # Flag for manual captures that should always narrate

            if jpeg_b64:
                # Capture current viewer state (includes layer sources automatically)
                state_json = {}
                try:
                    with ng_tracker.viewer.txn() as s:
                        state_json = s.to_json()

                    # DEBUG: Check if sources are in the state
                    if "layers" in state_json and len(state_json["layers"]) > 0:
                        first_layer = state_json["layers"][0]
                        has_source = "source" in first_layer
                        print(f"[STATE-DEBUG] First layer '{first_layer.get('name')}' has source: {has_source}", flush=True)
                        if has_source:
                            source_val = first_layer['source']
                            print(f"[STATE-DEBUG] Source type: {type(source_val)}", flush=True)
                            print(f"[STATE-DEBUG] Source value: {source_val}", flush=True)
                            # Also log full state JSON structure
                            import json as json_module
                            print(f"[STATE-DEBUG] Full state JSON (first 500 chars): {json_module.dumps(state_json)[:500]}", flush=True)
                        else:
                            print(f"[STATE-DEBUG] WARNING: No source in layer! Keys: {list(first_layer.keys())}", flush=True)

                    summary = ng_tracker.summarize_state(state_json)
                    ng_tracker.current_state_summary = summary
                    print(
                        f"[STATE] Updated: pos={summary.get('position', 'N/A')}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[STATE] Error getting state: {e}", flush=True)

                # Store the frame from client
                jpeg_bytes = base64.b64decode(jpeg_b64)

                ng_tracker.latest_frame = {
                    "jpeg_bytes": jpeg_bytes,
                    "jpeg_b64": jpeg_b64,
                    "timestamp": timestamp,
                    "state": state_json,  # CRITICAL: Use full state with sources, not summary!
                    "generate_audio": generate_audio,  # Store client's audio preference
                    "manual_capture": manual_capture,  # Flag to force narration generation
                }
                ng_tracker.latest_frame_ts = timestamp
                capture_type = "manual" if manual_capture else "auto"
                audio_setting = "enabled" if generate_audio else "disabled"
                print(f"[SCREENSHOT] Received from client: {len(jpeg_bytes)} bytes, type={capture_type}, voice={audio_setting}", flush=True)

                # Add to recording if active
                if recording_manager.is_recording:
                    print(f"[RECORDING] Adding frame to session (status: {recording_manager.current_session.status if recording_manager.current_session else 'no session'})", flush=True)
                    recording_manager.add_frame(
                        jpeg_bytes=jpeg_bytes,
                        state_json=state_json,
                        timestamp=timestamp
                    )

                return {"status": "ok"}
            else:
                return {"status": "error", "message": "No screenshot data"}
        except Exception as e:
            print(f"[ERROR] Failed to process client screenshot: {e}")
            return {"status": "error", "message": str(e)}

    @app.get("/api/debug-screenshot")
    async def debug_screenshot():
        """Return the latest screenshot for debugging."""
        latest_frame = ng_tracker.get_latest_frame()
        if latest_frame:
            # Save to file for inspection
            debug_path = Path("/tmp/ng_screenshot_debug.jpg")
            with open(debug_path, "wb") as f:
                f.write(base64.b64decode(latest_frame["jpeg_b64"]))
            return {
                "status": "ok",
                "size": len(latest_frame["jpeg_bytes"]),
                "timestamp": latest_frame["timestamp"],
                "saved_to": str(debug_path),
                "preview_url": f"data:image/jpeg;base64,{latest_frame['jpeg_b64'][:100]}...",
            }
        return {"status": "no_frames"}

    # Recording API endpoints
    @app.post("/api/recording/start")
    async def start_recording(request: Request):
        """Start a new recording session."""
        try:
            data = await request.json()
            fps = data.get("fps", 0.5)
            transition_type = data.get("transition_type", "cut")
            transition_duration = data.get("transition_duration", 0.5)

            session = recording_manager.start_recording(
                fps=fps,
                transition_type=transition_type,
                transition_duration=transition_duration
            )

            return {
                "status": "ok",
                "session_id": session.session_id,
                "message": "Recording started"
            }
        except Exception as e:
            print(f"[RECORDING] Start error: {e}", flush=True)
            return {
                "status": "error",
                "message": str(e)
            }

    @app.post("/api/recording/stop")
    async def stop_recording():
        """Stop the current recording session."""
        try:
            recording_manager.stop_recording()
            frame_count = len(recording_manager.current_session.frames) if recording_manager.current_session else 0
            return {
                "status": "ok",
                "message": "Recording stopped",
                "frame_count": frame_count
            }
        except Exception as e:
            print(f"[RECORDING] Stop error: {e}", flush=True)
            return {
                "status": "error",
                "message": str(e)
            }

    @app.get("/api/recording/status")
    async def get_recording_status():
        """Get current recording status."""
        if recording_manager.current_session:
            duration = 0
            if recording_manager.current_session.recording_started_at:
                if recording_manager.is_recording:
                    duration = (datetime.now() - recording_manager.current_session.recording_started_at).total_seconds()
                elif recording_manager.current_session.recording_stopped_at:
                    duration = (recording_manager.current_session.recording_stopped_at - recording_manager.current_session.recording_started_at).total_seconds()

            return {
                "is_recording": recording_manager.is_recording,
                "session_id": recording_manager.current_session.session_id,
                "status": recording_manager.current_session.status,
                "frame_count": len(recording_manager.current_session.frames),
                "duration": duration
            }
        return {
            "is_recording": False,
            "session_id": None,
            "status": "idle",
            "frame_count": 0,
            "duration": 0
        }

    async def compile_movie_task(session):
        """Background task for movie compilation."""
        try:
            print(f"[COMPILE] Starting compilation for session {session.session_id}", flush=True)

            # Run compilation in thread pool to avoid blocking
            compiler = MovieCompiler(session)

            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            output_file = await loop.run_in_executor(None, compiler.compile)

            print(f"[COMPILE] Completed: {output_file}", flush=True)

        except Exception as e:
            print(f"[COMPILE] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()

    @app.post("/api/recording/compile")
    async def compile_movie(request: Request, background_tasks: BackgroundTasks):
        """Compile the recorded frames into a movie."""
        try:
            if recording_manager.is_recording:
                return {
                    "status": "error",
                    "message": "Cannot compile while recording is in progress"
                }

            if not recording_manager.current_session:
                return {
                    "status": "error",
                    "message": "No recording session available"
                }

            # Get transition type from request body
            body = await request.json()
            transition_type = body.get("transition_type", "cut")

            # Update session transition type
            recording_manager.current_session.transition_type = transition_type
            recording_manager.current_session.save_metadata()

            print(f"[COMPILE] Using transition type: {transition_type}", flush=True)

            # Start compilation in background
            background_tasks.add_task(
                compile_movie_task,
                recording_manager.current_session
            )

            return {
                "status": "ok",
                "message": "Movie compilation started",
                "session_id": recording_manager.current_session.session_id
            }
        except Exception as e:
            print(f"[RECORDING] Compile error: {e}", flush=True)
            return {
                "status": "error",
                "message": str(e)
            }

    @app.get("/api/recording/compile/status/{session_id}")
    async def get_compile_status(session_id: str):
        """Get compilation status for a session."""
        # Check if current session matches
        if recording_manager.current_session and recording_manager.current_session.session_id == session_id:
            # Set progress based on status
            progress = 0.0
            if recording_manager.current_session.status == "compiling":
                # Show indeterminate progress (50%) while compiling
                progress = 0.5
            elif recording_manager.current_session.status == "completed":
                progress = 1.0

            return {
                "session_id": session_id,
                "status": recording_manager.current_session.status,
                "progress": progress,
                "error_message": recording_manager.current_session.error_message
            }
        return {
            "status": "error",
            "message": "Session not found"
        }

    @app.post("/api/narration/generate")
    async def generate_narration_for_screenshot(request: Request):
        """Generate narration for a specific screenshot (for movie creation)."""
        try:
            data = await request.json()
            jpeg_b64 = data.get("jpeg_b64")
            state = data.get("state")
            generate_audio = data.get("generate_audio", True)
            movie_mode = data.get("movie_mode", False)
            duration_seconds = data.get("duration_seconds", 25)
            existing_narration = data.get("existing_narration")  # For generating audio from existing text

            if not jpeg_b64:
                return {"status": "error", "message": "No screenshot data"}

            # If existing narration provided, use it; otherwise generate new narration
            if existing_narration:
                narration = existing_narration
                print(f"[NARRATION] Using existing narration text (skipping generation): {narration[:50]}...", flush=True)
            else:
                # Generate narration using the narrator
                print(f"[NARRATION] Generating new narration text...", flush=True)
                # For movie mode, use a special prompt that assumes specific duration
                if movie_mode:
                    # Create a custom prompt for movie narration
                    narration_prompt = f"""You are creating narration for a scientific tour movie.
This screenshot shows a Neuroglancer view of cellular structures.
This narration will be displayed for approximately {duration_seconds} seconds.

Create engaging, scientific narration that:
1. Describes what is visible in this view
2. Explains the scientific significance
3. Uses smooth transitions appropriate for a movie format
4. Is timed for about {duration_seconds} seconds of narration

Current view state: {json.dumps(state) if state else 'Not available'}

Provide narration:"""

                    narration = ng_tracker.narrator.generate_narration(
                        state,
                        screenshot_b64=jpeg_b64,
                        custom_prompt=narration_prompt
                    )
                else:
                    narration = ng_tracker.narrator.generate_narration(
                        state,
                        screenshot_b64=jpeg_b64
                    )

            # Generate audio if requested OR if in movie mode (always need audio for movies)
            audio_data = None
            should_generate_audio = narration and (generate_audio or movie_mode)
            if should_generate_audio:
                reason = "movie mode" if movie_mode and not generate_audio else "voice enabled"
                print(f"[NARRATION] Generating audio for narration (length: {len(narration)} chars, reason: {reason})...", flush=True)
                audio_data = await ng_tracker.narrator.generate_audio_async(narration)
                if audio_data:
                    print(f"[NARRATION] Audio generated successfully ({len(audio_data)} bytes)", flush=True)
                elif movie_mode:
                    print(f"[NARRATION] Warning: Failed to generate audio in movie mode", flush=True)
            elif narration and not generate_audio:
                print(f"[NARRATION] Skipping audio generation (voice disabled by user)", flush=True)

            return {
                "status": "ok",
                "narration": narration,
                "audio": audio_data,
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"[NARRATION] Generation error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e)
            }

    @app.post("/api/movie/create-from-screenshots")
    async def create_movie_from_screenshots(request: Request, background_tasks: BackgroundTasks):
        """Create a movie from selected screenshots with narrations."""
        try:
            data = await request.json()
            screenshots = data.get("screenshots", [])
            transition_type = data.get("transition_type", "cut")
            transition_duration = data.get("transition_duration", 2.0)

            if not screenshots:
                return {
                    "status": "error",
                    "message": "No screenshots provided"
                }

            # Create a new recording session from screenshots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"session_selected_{timestamp}"
            session_dir = recording_manager.base_recordings_dir / session_id

            # Create directory structure
            session_dir.mkdir(exist_ok=True)
            (session_dir / "frames").mkdir(exist_ok=True)
            (session_dir / "audio").mkdir(exist_ok=True)
            (session_dir / "states").mkdir(exist_ok=True)
            (session_dir / "urls").mkdir(exist_ok=True)
            (session_dir / "output").mkdir(exist_ok=True)

            # Create session object
            from recording import RecordingSession, FrameRecord
            session = RecordingSession(
                session_id=session_id,
                base_dir=session_dir,
                created_at=datetime.now(),
                fps=0.5,  # Doesn't matter for selected screenshots
                status="stopped",  # Skip recording state
                transition_type=transition_type,
                transition_duration=transition_duration
            )

            # Save each screenshot
            for i, screenshot in enumerate(screenshots):
                frame_number = i + 1

                # Save JPEG
                frame_file = f"frames/frame_{frame_number:05d}.jpg"
                jpeg_bytes = base64.b64decode(screenshot["jpeg_b64"])
                with open(session_dir / frame_file, "wb") as f:
                    f.write(jpeg_bytes)

                # Save state if available, otherwise save empty state
                state_file = f"states/state_{frame_number:05d}.json"
                if screenshot.get("state"):
                    # Check if state has layer sources
                    state_has_sources = False
                    if "layers" in screenshot["state"]:
                        state_has_sources = any("source" in layer for layer in screenshot["state"]["layers"])

                    with open(session_dir / state_file, "w") as f:
                        json.dump(screenshot["state"], f, indent=2)
                    print(f"[MOVIE] Frame {frame_number} state saved (has_sources={state_has_sources}, layers={len(screenshot['state'].get('layers', []))})", flush=True)
                else:
                    # Create empty state placeholder
                    with open(session_dir / state_file, "w") as f:
                        json.dump({}, f, indent=2)
                    print(f"[MOVIE] Warning: Frame {frame_number} has no state, using empty state", flush=True)

                # Save URL (generate from state)
                urls_file = f"urls/url_{frame_number:05d}.txt"
                if screenshot.get("state"):
                    from recording import generate_public_url
                    url = generate_public_url(screenshot["state"])
                    with open(session_dir / urls_file, "w") as f:
                        f.write(url)
                else:
                    # Create placeholder URL file
                    with open(session_dir / urls_file, "w") as f:
                        f.write("No state available for this frame\n")

                # Save audio if available and calculate duration
                narration_file = None
                display_duration = 2.0  # Default duration
                if screenshot.get("audio"):
                    narration_file = f"audio/narration_{frame_number:05d}.mp3"
                    audio_bytes = base64.b64decode(screenshot["audio"])
                    audio_path = session_dir / narration_file
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)

                    # Calculate actual audio duration using ffprobe
                    display_duration = recording_manager._get_audio_duration(audio_path)
                    print(f"[MOVIE] Frame {frame_number} audio duration: {display_duration:.2f}s", flush=True)

                # Create frame record
                frame_record = FrameRecord(
                    frame_number=frame_number,
                    timestamp=screenshot.get("timestamp", time.time()),
                    frame_file=frame_file,
                    state_file=state_file,
                    urls_file=urls_file,
                    has_narration=bool(screenshot.get("narration")),
                    narration_file=narration_file,
                    narration_text=screenshot.get("narration"),
                    display_duration=display_duration
                )

                session.frames.append(frame_record)

            # Save metadata
            session.save_metadata()

            # Set as current session and start compilation
            recording_manager.current_session = session

            # Start compilation in background
            background_tasks.add_task(
                compile_movie_task,
                session
            )

            return {
                "status": "ok",
                "message": "Movie compilation started from selected screenshots",
                "session_id": session_id,
                "frame_count": len(screenshots)
            }

        except Exception as e:
            print(f"[MOVIE] Creation error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e)
            }

    # Query Mode API endpoints
    @app.get("/api/mode")
    async def get_mode():
        """Get current mode (explore or query)."""
        return {"mode": current_mode["mode"]}

    @app.post("/api/mode/set")
    async def set_mode(request: Request):
        """Switch between explore, query, and analysis modes."""
        try:
            data = await request.json()
            mode = data.get("mode", "explore")

            if mode not in ["explore", "query", "analysis"]:
                return {"status": "error", "message": "Invalid mode. Must be 'explore', 'query', or 'analysis'."}

            current_mode["mode"] = mode
            print(f"[MODE] Switched to {mode} mode", flush=True)

            # Note: Narration is controlled by client's voice toggle (generate_audio flag)
            # Server-side narration_enabled is kept for backward compatibility but
            # actual audio generation is controlled by the client's generate_audio preference

            # Broadcast mode change to all connected clients
            message = {
                "type": "mode_change",
                "mode": mode,
                "timestamp": time.time()
            }
            disconnected = set()
            for connection in active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)
            active_connections.difference_update(disconnected)

            return {"status": "ok", "mode": mode}

        except Exception as e:
            print(f"[MODE] Error setting mode: {e}", flush=True)
            return {"status": "error", "message": str(e)}

    @app.post("/api/query/ask")
    async def process_query(request: Request):
        """Process natural language query using AI agent."""
        if not app.state.query_agent:
            return {
                "status": "error",
                "message": "Query mode not available. No database configured (check ORGANELLE_CSV_PATHS in .env)."
            }

        try:
            data = await request.json()
            user_query = data.get("query", "").strip()
            generate_audio = data.get("generate_audio", True)  # Default to True for backward compatibility

            if not user_query:
                return {"status": "error", "message": "Empty query"}

            print(f"[QUERY] Processing: {user_query} (audio: {'yes' if generate_audio else 'no'})", flush=True)

            # Process query with QueryAgent
            result = app.state.query_agent.process_query(user_query)

            # If navigation query, update Neuroglancer state
            if result.get("type") == "navigation" and result.get("navigation"):
                nav = result["navigation"]
                try:
                    with ng_tracker.viewer.txn() as s:
                        s.position = nav["position"]
                        if "scale" in nav:
                            s.crossSectionScale = nav["scale"]
                    print(f"[QUERY] Navigated to position {nav['position']} with scale {nav.get('scale')}", flush=True)
                except Exception as e:
                    print(f"[QUERY] Navigation failed: {e}", flush=True)
                    result["navigation_error"] = str(e)

            # If visualization query, update segment visibility
            if result.get("type") == "visualization" and result.get("visualization"):
                viz = result["visualization"]

                # Handle both single command (dict) and multiple commands (list)
                viz_commands = viz if isinstance(viz, list) else [viz]

                try:
                    with ng_tracker.viewer.txn() as s:
                        for viz_cmd in viz_commands:
                            layer_name = viz_cmd["layer_name"]
                            segment_ids = viz_cmd["segment_ids"]
                            action = viz_cmd.get("action", "show_only")

                            # Get the layer
                            if layer_name not in s.layers:
                                print(f"[QUERY] Warning: Layer '{layer_name}' not found, skipping", flush=True)
                                continue

                            layer = s.layers[layer_name]

                            # Convert segment IDs to integers (Neuroglancer expects uint64)
                            try:
                                segment_ids_int = [int(sid) for sid in segment_ids]
                            except (ValueError, TypeError) as e:
                                print(f"[QUERY] Warning: Could not convert segment IDs to integers: {e}", flush=True)
                                segment_ids_int = segment_ids

                            # Update segment visibility based on action
                            if action == "show_only":
                                # Clear existing segments and show only these
                                layer.segments = set(segment_ids_int)
                            elif action == "add":
                                # Add to existing visible segments
                                current_segments = set(layer.segments) if hasattr(layer, 'segments') else set()
                                layer.segments = current_segments | set(segment_ids_int)
                            elif action == "remove":
                                # Remove from visible segments
                                current_segments = set(layer.segments) if hasattr(layer, 'segments') else set()
                                layer.segments = current_segments - set(segment_ids_int)

                            print(f"[QUERY] Updated layer '{layer_name}' with {len(segment_ids)} segments (action: {action})", flush=True)

                    if len(viz_commands) > 1:
                        print(f"[QUERY] Applied {len(viz_commands)} visualization commands", flush=True)

                except Exception as e:
                    print(f"[QUERY] Visualization failed: {e}", flush=True)
                    result["visualization_error"] = str(e)

            # Generate audio for response (reuse TTS pipeline) if requested
            audio_data = None
            if generate_audio and result.get("answer"):
                try:
                    print(f"[QUERY] Generating audio for answer (voice enabled)...", flush=True)
                    audio_data = await ng_tracker.narrator.generate_audio_async(result["answer"])
                    print(f"[QUERY] Generated audio: {len(audio_data) if audio_data else 0} bytes", flush=True)
                except Exception as e:
                    print(f"[QUERY] Audio generation failed: {e}", flush=True)
            elif not generate_audio and result.get("answer"):
                print(f"[QUERY] Audio generation skipped (voice disabled by user)", flush=True)

            return {
                "status": "ok",
                "result": result,
                "audio": audio_data,
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"[QUERY] Error processing query: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Query processing failed: {str(e)}"
            }

    @app.post("/api/analysis/ask")
    async def process_analysis(request: Request):
        """Process analysis request with AI code generation and container execution."""
        if not app.state.analysis_agent or not app.state.container_sandbox:
            return {
                "status": "error",
                "message": "Analysis mode not available. Check container runtime (Docker/Apptainer) and database configuration."
            }

        try:
            data = await request.json()
            user_query = data.get("query", "").strip()

            if not user_query:
                return {"status": "error", "message": "Empty query"}

            print(f"[ANALYSIS] Processing: {user_query}", flush=True)

            # Generate code using AI with timing
            code_gen_start = time.time()
            code_result = app.state.analysis_agent.generate_analysis_code(user_query)
            code_gen_time = time.time() - code_gen_start

            if "error" in code_result:
                return {
                    "status": "error",
                    "message": code_result["error"]
                }

            # Execute code in container sandbox
            session_id = f"session_{int(time.time()*1000)}"
            exec_result = app.state.container_sandbox.execute(
                code=code_result["code"],
                session_id=session_id
            )

            # Save metadata with timing information
            if hasattr(app.state, 'analysis_results_manager'):
                app.state.analysis_results_manager.save_session_metadata(
                    session_id=session_id,
                    query=user_query,
                    code=code_result["code"],
                    execution_result=exec_result,
                    code_generation_time=code_gen_time
                )

            # Return comprehensive response
            return {
                "status": "ok" if exec_result["status"] == "success" else "error",
                "query": user_query,
                "code": code_result["code"],
                "stdout": exec_result["stdout"],
                "stderr": exec_result["stderr"],
                "plots": exec_result["plots"],
                "output_path": exec_result["output_path"],
                "session_id": session_id,
                "timing": {
                    "code_generation_seconds": round(code_gen_time, 3),
                    "execution_seconds": round(exec_result["execution_time"], 3),
                    "total_seconds": round(code_gen_time + exec_result["execution_time"], 3)
                },
                "execution_status": exec_result["status"],
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"[ANALYSIS] Error processing analysis: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            }

    @app.get("/api/analysis/plot/{session_id}/{filename}")
    async def get_analysis_plot(session_id: str, filename: str):
        """Serve generated plot files with security validation."""
        # Validate inputs (prevent path traversal)
        if not session_id.startswith("session_") or ".." in filename or "/" in filename:
            return Response(status_code=400, content="Invalid request")

        file_path = Path(f"analysis_results/{session_id}/{filename}")

        if not file_path.exists():
            return Response(status_code=404, content="Plot file not found")

        # Determine media type based on file extension
        media_type_map = {
            ".html": "text/html",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml"
        }
        media_type = media_type_map.get(file_path.suffix, "application/octet-stream")

        return FileResponse(path=str(file_path), media_type=media_type)

    @app.get("/api/analysis/sessions")
    async def list_analysis_sessions(limit: int = 50):
        """
        List recent analysis sessions with summary information.

        Args:
            limit: Maximum number of sessions to return (default: 50)

        Returns:
            List of session summaries sorted by creation time (newest first)
        """
        if not hasattr(app.state, 'analysis_results_manager'):
            return {
                "status": "error",
                "message": "Analysis results manager not available"
            }

        try:
            sessions = app.state.analysis_results_manager.list_sessions(limit=limit)
            return {
                "status": "ok",
                "sessions": sessions,
                "total": len(sessions)
            }
        except Exception as e:
            print(f"[ANALYSIS] Error listing sessions: {e}", flush=True)
            return {
                "status": "error",
                "message": f"Failed to list sessions: {str(e)}"
            }

    @app.get("/api/analysis/session/{session_id}")
    async def get_analysis_session(session_id: str):
        """
        Get detailed information about a specific analysis session.

        Args:
            session_id: Session identifier

        Returns:
            Full session metadata including code, results, plots, and timing
        """
        # Validate input
        if not session_id.startswith("session_") or ".." in session_id or "/" in session_id:
            return Response(status_code=400, content="Invalid session ID")

        if not hasattr(app.state, 'analysis_results_manager'):
            return {
                "status": "error",
                "message": "Analysis results manager not available"
            }

        try:
            session_details = app.state.analysis_results_manager.get_session_details(session_id)

            if session_details is None:
                return Response(status_code=404, content="Session not found")

            return {
                "status": "ok",
                "session": session_details
            }
        except Exception as e:
            print(f"[ANALYSIS] Error getting session {session_id}: {e}", flush=True)
            return {
                "status": "error",
                "message": f"Failed to get session: {str(e)}"
            }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for streaming frames."""
        await websocket.accept()
        active_connections.add(websocket)
        print(f"[WS] Client connected. Total clients: {len(active_connections)}")

        last_sent_timestamp = None

        try:
            # Send initial frame immediately if available
            latest_frame = ng_tracker.get_latest_frame()
            if latest_frame:
                # Verify state has layer sources
                state_has_sources = False
                if latest_frame["state"] and "layers" in latest_frame["state"]:
                    state_has_sources = any("source" in layer for layer in latest_frame["state"]["layers"])

                message = {
                    "type": "frame",
                    "ts": latest_frame["timestamp"],
                    "jpeg_b64": latest_frame["jpeg_b64"],
                    "state": latest_frame["state"],
                }
                await websocket.send_json(message)
                last_sent_timestamp = latest_frame["timestamp"]
                print(f"[WS] Sent initial frame (state_has_sources={state_has_sources})", flush=True)

            # Keep connection alive and send updates
            while True:
                # Check for new frames every 200ms
                await asyncio.sleep(0.2)

                # Update viewer state before checking latest frame
                try:
                    with ng_tracker.viewer.txn() as s:
                        state_json = s.to_json()
                    ng_tracker.current_state_summary = ng_tracker.summarize_state(
                        state_json
                    )
                except Exception as e:
                    pass  # Silently skip state update errors

                latest_frame = ng_tracker.get_latest_frame()
                if latest_frame:
                    # Don't overwrite the full state! The frame already has the full state with sources
                    # from the screenshot endpoint (line 313). We only use current_state_summary for
                    # narration prompts, not for storing/sending to client.
                    # latest_frame["state"] is already set with full state including layer sources

                    # Check if we should generate narration (pass screenshot timestamp)
                    # Skip narration if disabled (e.g., during query-based navigation)
                    # For manual captures, always generate narration (bypass should_narrate check)
                    manual_capture = latest_frame.get("manual_capture", False)
                    should_generate_narration = ng_tracker.narration_enabled and (
                        manual_capture or ng_tracker.narrator.should_narrate(
                            ng_tracker.current_state_summary,
                            screenshot_ts=latest_frame["timestamp"],
                        )
                    )

                    if should_generate_narration:
                        manual_flag = " (manual)" if manual_capture else ""
                        print(f"[NARRATOR] Generating narration{manual_flag}...", flush=True)
                        narration = ng_tracker.narrator.generate_narration(
                            ng_tracker.current_state_summary,
                            screenshot_b64=latest_frame["jpeg_b64"],
                            screenshot_ts=latest_frame["timestamp"],
                        )
                        if narration:
                            # Generate audio only if client requested it
                            audio_data = None
                            client_wants_audio = latest_frame.get("generate_audio", True)
                            if client_wants_audio:
                                print(f"[NARRATOR] Generating audio (voice enabled)...", flush=True)
                                audio_data = await ng_tracker.narrator.generate_audio_async(
                                    narration
                                )
                            else:
                                print(f"[NARRATOR] Skipping audio generation (voice disabled by user)", flush=True)

                            # Save audio to recording if active
                            if recording_manager.is_recording and audio_data:
                                # Decode base64 audio to bytes
                                audio_bytes = base64.b64decode(audio_data)

                                # Determine audio format based on TTS engine
                                audio_format = "mp3"  # default for edge-tts
                                if hasattr(ng_tracker.narrator, 'tts_engine'):
                                    if ng_tracker.narrator.tts_engine in ["coqui", "chatterbox"]:
                                        audio_format = "wav"

                                # Update the most recent frame with narration
                                if recording_manager.current_session and recording_manager.current_session.frames:
                                    frame_number = len(recording_manager.current_session.frames)
                                    recording_manager.update_frame_narration(
                                        frame_number=frame_number,
                                        narration_text=narration,
                                        audio_bytes=audio_bytes,
                                        audio_format=audio_format
                                    )

                            # Send narration message
                            narration_msg = {
                                "type": "narration",
                                "text": narration,
                                "timestamp": time.time(),
                                "audio": audio_data,  # base64 MP3 or WAV or None
                            }
                            await websocket.send_json(narration_msg)
                            print(
                                f"[NARRATOR] Sent narration with audio: {len(audio_data) if audio_data else 0} bytes",
                                flush=True,
                            )

                            # Clear manual_capture flag after narration is generated
                            # to prevent continuous generation on the same screenshot
                            if manual_capture:
                                latest_frame["manual_capture"] = False
                                print(f"[NARRATOR] Cleared manual_capture flag for this screenshot", flush=True)

                    # Only send if this is a new frame
                    if (
                        last_sent_timestamp is None
                        or latest_frame["timestamp"] > last_sent_timestamp
                    ):
                        # Verify state has layer sources before sending
                        state_has_sources = False
                        if latest_frame["state"] and "layers" in latest_frame["state"]:
                            state_has_sources = any("source" in layer for layer in latest_frame["state"]["layers"])

                        message = {
                            "type": "frame",
                            "ts": latest_frame["timestamp"],
                            "jpeg_b64": latest_frame["jpeg_b64"],
                            "state": latest_frame["state"],
                        }
                        await websocket.send_json(message)
                        last_sent_timestamp = latest_frame["timestamp"]
                        print(
                            f"[WS] Sent new frame (ts={latest_frame['timestamp']:.2f}, audio_gen={'yes' if latest_frame.get('generate_audio', True) else 'no'}, state_has_sources={state_has_sources})"
                        )

        except WebSocketDisconnect:
            print(f"[WS] Client disconnected")
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            active_connections.discard(websocket)
            print(f"[WS] Client removed. Total clients: {len(active_connections)}")

    # Mount static files for JS/CSS with no-cache headers
    web_dir = Path(__file__).parent.parent / "web"

    # Custom StaticFiles class to add cache-control headers
    class NoCacheStaticFiles(StaticFiles):
        async def get_response(self, path, scope):
            response = await super().get_response(path, scope)
            if isinstance(response, Response):
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    app.mount("/static", NoCacheStaticFiles(directory=str(web_dir)), name="static")

    return app
