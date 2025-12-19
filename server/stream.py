"""
FastAPI server with WebSocket streaming.
Stage 3: WebSocket endpoint that streams frames + state to browser.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
import asyncio
import json
import time
from pathlib import Path
from typing import Set
import httpx

# Will be set by main.py
ng_tracker = None


def create_app(tracker) -> FastAPI:
    """Create FastAPI app with WebSocket streaming."""
    global ng_tracker
    ng_tracker = tracker

    app = FastAPI(title="Neuroglancer Live Stream")

    # Track active WebSocket connections
    active_connections: Set[WebSocket] = set()

    # Queue for frame updates
    frame_queue = asyncio.Queue()

    # Queue for narration updates (Stage 4)
    narration_queue = asyncio.Queue()

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

            if jpeg_b64:
                # Capture current viewer state
                try:
                    with ng_tracker.viewer.txn() as s:
                        state_json = s.to_json()
                    summary = ng_tracker.summarize_state(state_json)
                    ng_tracker.current_state_summary = summary
                    print(
                        f"[STATE] Updated: pos={summary.get('position', 'N/A')}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[STATE] Error getting state: {e}", flush=True)

                # Store the frame from client
                import base64

                jpeg_bytes = base64.b64decode(jpeg_b64)

                ng_tracker.latest_frame = {
                    "jpeg_bytes": jpeg_bytes,
                    "jpeg_b64": jpeg_b64,
                    "timestamp": timestamp,
                    "state": ng_tracker.current_state_summary,
                }
                ng_tracker.latest_frame_ts = timestamp
                print(f"[SCREENSHOT] Received from client: {len(jpeg_bytes)} bytes")

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
            import base64
            from pathlib import Path

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
                message = {
                    "type": "frame",
                    "ts": latest_frame["timestamp"],
                    "jpeg_b64": latest_frame["jpeg_b64"],
                    "state": latest_frame["state"],
                }
                await websocket.send_json(message)
                last_sent_timestamp = latest_frame["timestamp"]
                print(f"[WS] Sent initial frame")

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
                    # Update state in the frame
                    latest_frame["state"] = ng_tracker.current_state_summary

                    # Check if we should generate narration
                    if ng_tracker.narrator.should_narrate(
                        ng_tracker.current_state_summary
                    ):
                        print(f"[NARRATOR] Generating narration...", flush=True)
                        narration = ng_tracker.narrator.generate_narration(
                            ng_tracker.current_state_summary,
                            screenshot_b64=latest_frame["jpeg_b64"],
                        )
                        if narration:
                            # Send narration message
                            narration_msg = {
                                "type": "narration",
                                "text": narration,
                                "timestamp": time.time(),
                            }
                            await websocket.send_json(narration_msg)
                            print(f"[NARRATOR] Sent: {narration}", flush=True)

                    # Only send if this is a new frame
                    if (
                        last_sent_timestamp is None
                        or latest_frame["timestamp"] > last_sent_timestamp
                    ):
                        message = {
                            "type": "frame",
                            "ts": latest_frame["timestamp"],
                            "jpeg_b64": latest_frame["jpeg_b64"],
                            "state": latest_frame["state"],
                        }
                        await websocket.send_json(message)
                        last_sent_timestamp = latest_frame["timestamp"]
                        print(
                            f"[WS] Sent new frame (ts={latest_frame['timestamp']:.2f})"
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
