# Screenshot Solution: postMessage Communication

## The Problem

Browser security prevents parent pages from capturing iframe content:
- **Python Screenshot API**: Times out with embedded iframes
- **html2canvas**: Can only see the iframe container, not its rendered content
- **Proxy approach**: Breaks Neuroglancer's Server-Sent Events connection

## The Solution: postMessage Screenshot Capture

Instead of trying to capture the iframe from the outside, we have the **iframe capture itself** and send the screenshot to the parent via `postMessage`.

### Architecture

```
┌─────────────────────────────────────┐
│   Parent Page (app.js)              │
│                                     │
│   1. Send postMessage →             │
│      {type: 'captureScreenshot'}    │
│                                     │
│   3. Receive postMessage ←          │
│      {type: 'screenshot',           │
│       jpeg_b64: '...'}              │
│                                     │
│   4. Display + Send to Server       │
└─────────────────────────────────────┘
              ↕ postMessage
┌─────────────────────────────────────┐
│   Neuroglancer Iframe               │
│   (ng-screenshot-handler.js)        │
│                                     │
│   2. Capture from canvas →          │
│      document.querySelector('canvas')│
│      .toDataURL('image/jpeg')       │
└─────────────────────────────────────┘
```

### How It Works

1. **Parent page** (`app.js`) sends a `postMessage` to the iframe requesting a screenshot
2. **Iframe script** (`ng-screenshot-handler.js`) listens for this message
3. **Iframe script** finds the Neuroglancer canvas and uses `.toDataURL()` to capture it
4. **Iframe script** sends the screenshot back via `postMessage`
5. **Parent page** receives the screenshot, displays it, and forwards to server

### Key Benefits

✅ **Works with iframe isolation** - Captures from inside the iframe  
✅ **No cross-origin issues** - postMessage works across origins  
✅ **Doesn't break Neuroglancer** - No proxy for SSE needed  
✅ **Direct canvas capture** - Gets actual rendered WebGL content  

## Files Modified

### 1. `web/ng-screenshot-handler.js` (NEW)
Script that runs inside the Neuroglancer iframe to handle screenshot capture.

**Key functions:**
- Waits for Neuroglancer to load (detects canvas)
- Listens for `captureScreenshot` messages from parent
- Captures screenshot from Neuroglancer's canvas element
- Sends screenshot back via postMessage

### 2. `web/app.js`
Modified screenshot capture flow:

**Changes:**
- Removed `html2canvas` approach
- Added `setupMessageListener()` to receive screenshots from iframe
- Added `requestScreenshot()` to request screenshots via postMessage
- Added `handleScreenshotFromIframe()` to process received screenshots
- Added `injectScreenshotHandler()` to inject script into iframe (if same-origin)

### 3. `server/stream.py`
Added script injection to proxy:

**Changes:**
- Proxy now injects `ng-screenshot-handler.js` into Neuroglancer HTML responses
- Ensures the script loads even with cross-origin iframes

### 4. `web/index.html`
- Removed `html2canvas` dependency (no longer needed)

## Testing the Solution

### Step 1: Start the Server

```bash
cd /groups/cellmap/cellmap/ackermand/Programming/tourguide
conda activate igneous_daskified
python server/main.py
```

Expected output:
```
Neuroglancer viewer URL: http://...
FastAPI server starting on http://0.0.0.0:8090
```

### Step 2: Open the Web Interface

Navigate to: `http://localhost:8090/`

### Step 3: Check Browser Console

Open Developer Tools → Console. You should see:

```
[MESSAGE] Neuroglancer iframe is ready
[SCREENSHOT] Iframe not ready      ← Initial attempts while loading
[MESSAGE] Received screenshot from iframe
[SCREENSHOT] Sent to server
```

### Step 4: Verify Screenshots

1. **Live Screenshot panel** should show the Neuroglancer view
2. **Frame count** should increment (~1 fps)
3. **Server console** should show:
   ```
   [SCREENSHOT] Received from client: XXXX bytes
   ```

### Step 5: Navigate in Neuroglancer

Pan/zoom in the Neuroglancer viewer. Screenshots should update to show the new view.

## Debugging

### "Iframe not ready" messages

**Cause**: Script not loaded yet  
**Fix**: Wait 3-5 seconds after page load

### No screenshots appearing

**Check 1**: Browser console for errors  
**Check 2**: Check if proxy is injecting script:
```
# In browser network tab, look for HTML response
# Should contain: <script src="/static/ng-screenshot-handler.js"></script>
```

**Check 3**: Try requesting manually in console:
```javascript
document.getElementById('ng-iframe').contentWindow.postMessage({
    type: 'captureScreenshot', width: 800, height: 600
}, '*');
```

### Proxy errors

If you see `[PROXY ERROR]` in server console:
1. Check that Neuroglancer is running (should see URL printed on startup)
2. Check that port 9999 is accessible
3. Try accessing Neuroglancer directly: `http://localhost:9999/`

## Alternative: Direct Neuroglancer (No Proxy)

If the proxy approach causes issues, you can switch to direct Neuroglancer URL:

**In `server/stream.py`**, change:
```python
return {"url": f"/ng-proxy/{path}"}
```

To:
```python
return {"url": ng_tracker.get_url()}
```

**Trade-off**: This avoids proxy complexity but requires CORS headers or same-origin setup.

## Next Steps

Once screenshots are working:

1. **Tune screenshot rate**: Adjust `screenshotFps` in `app.js` (currently 1 fps)
2. **Add screenshot to narration**: Pass screenshots to AI narrator
3. **Optimize image size**: Reduce quality/resolution if needed
4. **Add screenshot history**: Store recent screenshots for comparison

## Why This Works

The key insight: **Browser security blocks external access to iframe rendering, but the iframe can always capture itself**.

By having the iframe do the work and communicate via `postMessage`, we bypass all the security restrictions that blocked previous approaches.
