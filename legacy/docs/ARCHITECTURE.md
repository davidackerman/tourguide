# Architecture: Screenshot Capture Flow

## Overview

```
┌────────────────────────────────────────────────────────────────┐
│                         Browser                                 │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Parent Page (http://localhost:8090)                     │  │
│  │  File: web/app.js                                        │  │
│  │                                                          │  │
│  │  1. Every 1 second:                                      │  │
│  │     requestScreenshot()                                  │  │
│  │     │                                                    │  │
│  │     ├─► postMessage ──────────────────────┐             │  │
│  │     │   {type: 'captureScreenshot'}       │             │  │
│  │     │                                      ↓             │  │
│  │     │   ┌────────────────────────────────────────────┐  │  │
│  │     │   │  Neuroglancer Iframe                       │  │  │
│  │     │   │  (Cross-origin isolated)                   │  │  │
│  │     │   │  File: web/ng-screenshot-handler.js        │  │  │
│  │     │   │                                            │  │  │
│  │     │   │  2. On message:                            │  │  │
│  │     │   │     Find canvas element                    │  │  │
│  │     │   │     const canvas = document.querySelector( │  │  │
│  │     │   │       'canvas.neuroglancer-gl-canvas'      │  │  │
│  │     │   │     );                                      │  │  │
│  │     │   │                                            │  │  │
│  │     │   │  3. Capture:                               │  │  │
│  │     │   │     canvas.toDataURL('image/jpeg', 0.8)    │  │  │
│  │     │   │                                            │  │  │
│  │     ┌───┤  4. Send back:                             │  │  │
│  │     │   │     postMessage ◄──────────────────────────┤  │  │
│  │     │   │     {type: 'screenshot',                   │  │  │
│  │     │   │      jpeg_b64: '...'}                      │  │  │
│  │     │   └────────────────────────────────────────────┘  │  │
│  │     │                                                    │  │
│  │     ├─► 5. handleScreenshotFromIframe(jpeg_b64)         │  │
│  │     │       - Display in <img> tag                      │  │
│  │     │       - Update frame count                        │  │
│  │     │       - Calculate FPS                             │  │
│  │     │                                                    │  │
│  │     └─► 6. sendScreenshotToServer(jpeg_b64)             │  │
│  │             POST /api/screenshot                         │  │
│  └───────────────────────────────│──────────────────────────┘  │
│                                  │                             │
└──────────────────────────────────┼─────────────────────────────┘
                                   │
                                   ↓ HTTP POST
┌──────────────────────────────────────────────────────────────┐
│                      Server (FastAPI)                         │
│                      File: server/stream.py                   │
│                                                               │
│  @app.post("/api/screenshot")                                │
│  async def receive_screenshot(request):                      │
│      7. Decode base64 → JPEG bytes                           │
│      8. Store in ng_tracker.latest_frame                     │
│      9. Broadcast via WebSocket to all clients               │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  State Tracker                                         │  │
│  │  File: server/ng.py                                    │  │
│  │                                                        │  │
│  │  10. latest_frame = {                                 │  │
│  │        'jpeg_bytes': ...,                             │  │
│  │        'jpeg_b64': ...,                               │  │
│  │        'state': current_state_summary,                │  │
│  │        'timestamp': ...                               │  │
│  │      }                                                 │  │
│  │                                                        │  │
│  │  11. AI Narrator can access:                          │  │
│  │      - Screenshot (visual)                            │  │
│  │      - State (position, layers, etc.)                 │  │
│  │      - Generate narration combining both              │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Screenshot Handler (web/ng-screenshot-handler.js)

**Purpose**: Runs inside Neuroglancer iframe to capture screenshots

**Key Functions**:
```javascript
// Wait for Neuroglancer to load
waitForNeuroglancer(callback)

// Capture screenshot from canvas
captureScreenshot(width, height)

// Listen for requests from parent
window.addEventListener('message', ...)

// Notify parent when ready
window.parent.postMessage({type: 'ready'}, '*')
```

**Security**: Can access iframe's own canvas (no cross-origin restrictions)

### 2. Parent Page Client (web/app.js)

**Purpose**: Request screenshots and manage display

**Key Methods**:
```javascript
// Setup message listener
setupMessageListener()

// Request screenshot from iframe
requestScreenshot()

// Handle received screenshot
handleScreenshotFromIframe(jpeg_b64)

// Send to server
sendScreenshotToServer(jpeg_b64)
```

**Timing**: Requests screenshots at configurable FPS (default 1 fps)

### 3. Proxy Server (server/stream.py)

**Purpose**: Inject screenshot handler into Neuroglancer HTML

**Key Functions**:
```python
# Proxy Neuroglancer requests
@app.api_route("/ng-proxy/{path:path}", ...)
async def neuroglancer_proxy(path, request):
    # Get response from Neuroglancer
    response = await client.request(...)
    
    # Inject script if HTML
    if 'text/html' in content_type:
        html = html.replace('</head>', 
            '<script src="/static/ng-screenshot-handler.js"></script></head>')
        
    return Response(content=html, ...)
```

**Critical**: Updates Content-Length header after injection

### 4. Screenshot Receiver (server/stream.py)

**Purpose**: Receive and store screenshots from browser

```python
@app.post("/api/screenshot")
async def receive_screenshot(request):
    data = await request.json()
    jpeg_b64 = data.get('jpeg_b64')
    
    # Decode and store
    jpeg_bytes = base64.b64decode(jpeg_b64)
    ng_tracker.latest_frame = {
        'jpeg_bytes': jpeg_bytes,
        'jpeg_b64': jpeg_b64,
        'timestamp': timestamp,
        'state': ng_tracker.current_state_summary
    }
```

## Data Flow

### Screenshot Data

```
Neuroglancer Canvas (WebGL)
    ↓ toDataURL()
Base64 JPEG String (~30-50 KB)
    ↓ postMessage
Parent Page JavaScript
    ↓ Display in <img>
    ↓ HTTP POST
FastAPI Server
    ↓ Base64 decode
JPEG Bytes
    ↓ Store in memory
AI Narrator (future)
```

### State Data

```
Neuroglancer State Change
    ↓ State callback
State Tracker (ng.py)
    ↓ Summarize
State Summary {position, scale, layers, ...}
    ↓ Attach to screenshot
Combined Frame Data
    ↓ WebSocket
Browser Display
```

## Message Protocol

### Parent → Iframe

```javascript
{
    type: 'captureScreenshot',
    width: 800,    // Optional
    height: 600    // Optional
}
```

### Iframe → Parent

```javascript
// Ready notification
{
    type: 'ready'
}

// Screenshot response
{
    type: 'screenshot',
    jpeg_b64: 'base64-encoded-jpeg-data',
    timestamp: 1234567890
}
```

### Browser → Server

```json
{
    "jpeg_b64": "base64-encoded-jpeg-data",
    "timestamp": 1234567890.123
}
```

## Security Considerations

### postMessage Security

**Current**: Uses `'*'` for origin (any origin accepted)

**Production**: Should validate origin:
```javascript
if (event.origin !== 'https://expected-domain.com') return;
```

### Canvas Access

**Why it works**: Script runs in same context as canvas
**Blocked**: External scripts trying to access cross-origin canvas

### Content Security Policy

May need to allow:
- `script-src 'self'` - Load our injected script
- `connect-src 'self'` - WebSocket connections
- `img-src data:` - Display base64 images

## Performance

### Screenshot Rate

- **Default**: 1 fps
- **Adjustable**: Change `screenshotFps` in app.js
- **Recommendation**: 1-2 fps for live monitoring

### Screenshot Size

- **Typical**: 20-50 KB per frame (JPEG, quality 0.8)
- **At 1 fps**: ~3 MB/minute bandwidth
- **At 2 fps**: ~6 MB/minute bandwidth

### Optimization Options

1. **Lower quality**: `toDataURL('image/jpeg', 0.6)`
2. **Smaller resolution**: Resize canvas before capture
3. **Skip unchanged frames**: Compare to previous screenshot
4. **Delta encoding**: Only send changed regions

## Extension Points

### Add Vision AI

```python
# In server/narrator.py
def generate_narration(self, summary, screenshot_b64):
    # Send to Claude, GPT-4V, etc.
    response = anthropic.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": screenshot_b64
                    }
                },
                {
                    "type": "text",
                    "text": f"What's in this brain region? Position: {summary['position']}"
                }
            ]
        }]
    )
```

### Add Screenshot History

```javascript
// In app.js
this.screenshotHistory = [];
handleScreenshotFromIframe(jpeg_b64) {
    this.screenshotHistory.push({
        jpeg_b64: jpeg_b64,
        timestamp: Date.now(),
        state: this.currentState
    });
    // Keep last 10
    if (this.screenshotHistory.length > 10) {
        this.screenshotHistory.shift();
    }
}
```

### Add Screenshot Recording

```javascript
// Record screenshots to video
const mediaRecorder = new MediaRecorder(canvas.captureStream(30));
mediaRecorder.start();
```

## Troubleshooting

### No screenshots?

1. Check browser console: Should see `[MESSAGE] Neuroglancer iframe is ready`
2. Check server logs: Should see `[PROXY] Injected screenshot handler`
3. Verify canvas exists: In iframe console, `document.querySelector('canvas')`

### Low FPS?

1. Increase `screenshotFps` in app.js
2. Check network latency (screenshot upload may be slow)
3. Reduce screenshot quality/size

### High CPU usage?

1. Decrease screenshot rate
2. Add frame skip logic
3. Only capture on state changes
