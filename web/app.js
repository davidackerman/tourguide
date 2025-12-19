/**
 * WebSocket client for Neuroglancer live stream
 */

class NGLiveStream {
    constructor() {
        this.ws = null;
        this.frameCount = 0;
        this.lastFrameTime = null;
        this.frameTimestamps = [];
        this.narrations = [];  // Keep history of narrations
        this.maxNarrations = 10;  // Maximum narrations to display

        // Screenshot capture settings
        this.screenshotInterval = null;
        this.screenshotFps = 0.5;  // Default 0.5 fps (1 screenshot every 2 seconds)
        this.lastScreenshotTime = 0;
        this.iframeReady = false;
        this.iframeReady = false;

        // DOM elements
        this.ngIframe = document.getElementById('ng-iframe');
        this.frameImg = document.getElementById('frame');
        this.frameOverlay = document.getElementById('frame-overlay');
        this.stateInfo = document.getElementById('state-info');
        this.statusText = document.getElementById('status-text');
        this.statusIndicator = document.querySelector('.status-indicator');
        this.frameCountEl = document.getElementById('frame-count');
        this.lastUpdateEl = document.getElementById('last-update');
        this.fpsEl = document.getElementById('fps');
        this.narrationContainer = document.getElementById('narration-container');
        this.screenshotFpsInput = document.getElementById('screenshot-fps');

        // Setup screenshot FPS control
        if (this.screenshotFpsInput) {
            this.screenshotFpsInput.addEventListener('change', () => {
                const newFps = parseFloat(this.screenshotFpsInput.value);
                if (newFps > 0 && newFps <= 5) {
                    this.screenshotFps = newFps;
                    console.log(`[SCREENSHOT] FPS changed to ${this.screenshotFps}`);
                    // Restart capture with new rate
                    this.restartScreenshotCapture();
                }
            });
        }

        this.loadNeuroglancerURL();
        this.connect();
        // Re-enable client-side screenshot capture (no cross-origin data now)
        this.setupPageScreenshotCapture();
    }

    setupMessageListener() {
        window.addEventListener('message', (event) => {
            // Security: verify message origin if needed
            // if (event.origin !== 'expected-origin') return;

            if (event.data && event.data.type === 'screenshot') {
                console.log('Received screenshot from iframe:', event.data.jpeg_b64?.substring(0, 50) + '...');
                this.handleScreenshotFromIframe(event.data.jpeg_b64);
            } else if (event.data && event.data.type === 'ready') {
                console.log('Neuroglancer iframe is ready');
                this.iframeReady = true;
            }
        });
    }

    handleScreenshotFromIframe(jpeg_b64) {
        if (!jpeg_b64) {
            console.error('No screenshot data received');
            return;
        }

        // Display the screenshot
        this.frameImg.src = `data:image/jpeg;base64,${jpeg_b64}`;
        this.frameOverlay.style.display = 'none';

        // Update frame count
        this.frameCount++;
        this.frameCountEl.textContent = this.frameCount;

        // Update timestamp
        const now = Date.now();
        this.lastFrameTime = now;
        this.lastUpdateEl.textContent = new Date().toLocaleTimeString();

        // Calculate FPS
        this.frameTimestamps.push(now);
        if (this.frameTimestamps.length > 30) {
            this.frameTimestamps.shift();
        }
        this.updateFPS();

        // Send to server for processing
        this.sendScreenshotToServer(jpeg_b64);
    }

    async sendScreenshotToServer(jpeg_b64) {
        try {
            await fetch('/api/screenshot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    jpeg_b64: jpeg_b64,
                    timestamp: Date.now() / 1000
                })
            });
        } catch (e) {
            console.error('Failed to send screenshot to server:', e);
        }
    }

    async loadNeuroglancerURL() {
        try {
            const response = await fetch('/api/ng-url');
            const data = await response.json();
            // Use proxied URL for same-origin iframe embedding
            this.ngIframe.src = data.url;
            console.log('Neuroglancer iframe URL loaded:', data.url);
        } catch (e) {
            console.error('Failed to load Neuroglancer URL:', e);
        }
    }

    injectScreenshotHandler() {
        try {
            // For same-origin iframes, we can inject directly
            const iframeDoc = this.ngIframe.contentDocument || this.ngIframe.contentWindow.document;

            if (iframeDoc) {
                // Same-origin: inject directly
                const script = iframeDoc.createElement('script');
                script.textContent = `
                    // Neuroglancer Screenshot Handler (injected)
                    (function() {
                        console.log('[NG-SCREENSHOT] Handler initialized');

                        function waitForNeuroglancer(callback) {
                            const checkInterval = setInterval(() => {
                                if (document.querySelector('canvas')) {
                                    clearInterval(checkInterval);
                                    console.log('[NG-SCREENSHOT] Canvas detected');
                                    callback();
                                }
                            }, 500);
                            setTimeout(() => clearInterval(checkInterval), 30000);
                        }

                        function captureScreenshot(width, height) {
                            try {
                                const canvas = document.querySelector('canvas.neuroglancer-gl-canvas') ||
                                              document.querySelector('canvas');
                                if (!canvas) {
                                    console.error('[NG-SCREENSHOT] No canvas found');
                                    return null;
                                }

                                const tempCanvas = document.createElement('canvas');
                                const ctx = tempCanvas.getContext('2d');
                                tempCanvas.width = width || canvas.width;
                                tempCanvas.height = height || canvas.height;
                                ctx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);
                                const jpeg_b64 = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                                console.log('[NG-SCREENSHOT] Captured:', jpeg_b64.length, 'bytes');
                                return jpeg_b64;
                            } catch (error) {
                                console.error('[NG-SCREENSHOT] Error:', error);
                                return null;
                            }
                        }

                        window.addEventListener('message', (event) => {
                            if (event.data && event.data.type === 'captureScreenshot') {
                                console.log('[NG-SCREENSHOT] Received capture request');
                                const jpeg_b64 = captureScreenshot(event.data.width || 800, event.data.height || 600);
                                if (jpeg_b64) {
                                    window.parent.postMessage({
                                        type: 'screenshot',
                                        jpeg_b64: jpeg_b64,
                                        timestamp: Date.now()
                                    }, '*');
                                    console.log('[NG-SCREENSHOT] Sent to parent');
                                }
                            }
                        });

                        waitForNeuroglancer(() => {
                            window.parent.postMessage({ type: 'ready' }, '*');
                            console.log('[NG-SCREENSHOT] Ready');
                        });
                    })();
                `;
                iframeDoc.head.appendChild(script);
                console.log('[INJECT] Screenshot handler injected directly');
            } else {
                // Cross-origin: Can't inject directly
                console.log('[INJECT] Cross-origin iframe - script injection not possible');
                console.log('[INJECT] Screenshot capture will not work with cross-origin Neuroglancer');
                // Mark as ready anyway
                setTimeout(() => {
                    this.iframeReady = true;
                }, 2000);
            }

        } catch (error) {
            console.error('[INJECT] Error:', error);
            console.log('[INJECT] Will rely on user opening Neuroglancer in same origin');
            setTimeout(() => {
                this.iframeReady = true;
            }, 2000);
        }
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log('Connecting to:', wsUrl);
        this.updateStatus('connecting', 'Connecting...');

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('connected', 'Connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('error', 'Connection error');
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
            this.updateStatus('disconnected', 'Disconnected');

            // Attempt to reconnect after 2 seconds
            setTimeout(() => this.connect(), 2000);
        };
    }

    handleMessage(data) {
        if (data.type === 'frame') {
            this.handleFrame(data);
        } else if (data.type === 'narration') {
            this.handleNarration(data);
        }
    }

    handleFrame(data) {
        // Update frame image (if element exists)
        if (this.frameImg) {
            this.frameImg.src = `data:image/jpeg;base64,${data.jpeg_b64}`;
        }
        if (this.frameOverlay) {
            this.frameOverlay.style.display = 'none';
        }

        // Update frame count
        this.frameCount++;
        if (this.frameCountEl) {
            this.frameCountEl.textContent = this.frameCount;
        }

        // Update timestamp
        const now = Date.now();
        this.lastFrameTime = now;
        this.lastUpdateEl.textContent = new Date(data.ts * 1000).toLocaleTimeString();

        // Calculate FPS
        this.frameTimestamps.push(now);
        // Keep only last 30 frames
        if (this.frameTimestamps.length > 30) {
            this.frameTimestamps.shift();
        }
        this.updateFPS();

        // Update state information
        if (data.state) {
            this.updateState(data.state);
        }
    }

    updateFPS() {
        if (this.frameTimestamps.length < 2) {
            this.fpsEl.textContent = '0';
            return;
        }

        const timeSpan = this.frameTimestamps[this.frameTimestamps.length - 1] - this.frameTimestamps[0];
        const fps = ((this.frameTimestamps.length - 1) / timeSpan) * 1000;
        this.fpsEl.textContent = fps.toFixed(1);
    }

    updateState(state) {
        const parts = [];

        // Position
        if (state.position) {
            const pos = state.position;
            parts.push(`
                <div class="state-item">
                    <strong>Position:</strong>
                    <span class="mono">[${pos[0].toFixed(0)}, ${pos[1].toFixed(0)}, ${pos[2].toFixed(0)}]</span>
                </div>
            `);
        }

        // Scale
        if (state.scale !== undefined) {
            parts.push(`
                <div class="state-item">
                    <strong>Scale:</strong>
                    <span class="mono">${state.scale.toFixed(2)}</span>
                </div>
            `);
        }

        // Orientation
        if (state.orientation) {
            const orient = state.orientation;
            parts.push(`
                <div class="state-item">
                    <strong>Orientation:</strong>
                    <span class="mono">[${orient.map(v => v.toFixed(2)).join(', ')}]</span>
                </div>
            `);
        }

        // Layers
        if (state.layers && state.layers.length > 0) {
            const visibleLayers = state.layers.filter(l => l.visible);
            const layerList = visibleLayers.map(l =>
                `<span class="layer-tag">${l.name} (${l.type})</span>`
            ).join('');
            parts.push(`
                <div class="state-item">
                    <strong>Visible Layers:</strong>
                    <div class="layer-list">${layerList}</div>
                </div>
            `);
        }

        // Selected segments
        if (state.selected_segments && state.selected_segments.length > 0) {
            const segList = state.selected_segments.map(s =>
                `<span class="segment-tag">${s}</span>`
            ).join('');
            parts.push(`
                <div class="state-item">
                    <strong>Selected Segments:</strong>
                    <div class="segment-list">${segList}</div>
                </div>
            `);
        }

        if (parts.length > 0) {
            this.stateInfo.innerHTML = parts.join('');
        } else {
            this.stateInfo.innerHTML = '<p class="placeholder">No state information available</p>';
        }
    }

    handleNarration(data) {
        console.log('Narration received:', data.text);

        // Add to narrations list (most recent first)
        this.narrations.unshift({
            text: data.text,
            timestamp: data.timestamp || Date.now() / 1000
        });

        // Keep only the most recent narrations
        if (this.narrations.length > this.maxNarrations) {
            this.narrations = this.narrations.slice(0, this.maxNarrations);
        }

        // Update the narration display
        this.updateNarrationDisplay();
        
        // Queue audio if available
        if (data.audio) {
            this.queueAudio(data.audio);
        }
    }
    
    queueAudio(base64Audio) {
        // Initialize audio queue if needed
        if (!this.audioQueue) {
            this.audioQueue = [];
            this.isPlayingAudio = false;
        }
        
        // Add to queue
        this.audioQueue.push(base64Audio);
        
        // Start playing if not already playing
        if (!this.isPlayingAudio) {
            this.playNextAudio();
        }
    }
    
    playNextAudio() {
        // Check if there's audio to play
        if (this.audioQueue.length === 0) {
            this.isPlayingAudio = false;
            return;
        }
        
        this.isPlayingAudio = true;
        const base64Audio = this.audioQueue.shift();
        this.playAudio(base64Audio);
    }
    
    playAudio(base64Audio) {
        try {
            // Decode base64 audio
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            // Create blob and URL (MP3 format from edge-tts)
            const blob = new Blob([bytes], { type: 'audio/mpeg' });
            const audioUrl = URL.createObjectURL(blob);
            
            // Play audio
            const audio = new Audio(audioUrl);
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                // Play next audio in queue
                this.playNextAudio();
            };
            audio.onerror = (err) => {
                console.error('Audio playback failed:', err);
                URL.revokeObjectURL(audioUrl);
                // Continue to next audio even on error
                this.playNextAudio();
            };
            audio.play().catch(err => {
                console.error('Audio play error:', err);
                URL.revokeObjectURL(audioUrl);
                // Continue to next audio even on error
                this.playNextAudio();
            });
            
            console.log('Playing audio narration');
        } catch (err) {
            console.error('Audio decode failed:', err);
            // Continue to next audio even on error
            this.playNextAudio();
        }
    }

    updateNarrationDisplay() {
        if (this.narrations.length === 0) {
            this.narrationContainer.innerHTML = '<div class="narration-placeholder">Navigate around to trigger AI narration...</div>';
            return;
        }

        const narrationHTML = this.narrations.map((narration, index) => {
            const timeStr = new Date(narration.timestamp * 1000).toLocaleTimeString();
            const isLatest = index === 0;
            return `
                <div class="narration-item ${isLatest ? 'latest' : ''}">
                    <div class="narration-time">${timeStr}</div>
                    <div class="narration-text">${this.escapeHtml(narration.text)}</div>
                </div>
            `;
        }).join('');

        this.narrationContainer.innerHTML = narrationHTML;

        // Scroll to top to show latest narration
        this.narrationContainer.scrollTop = 0;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    updateStatus(status, text) {
        this.statusText.textContent = text;
        this.statusIndicator.className = `status-indicator status-${status}`;
    }

    setupViewerScreenshotCapture() {
        // Use modern browser screenshot API or fallback to canvas approach
        setTimeout(() => {
            setInterval(async () => {
                await this.captureViewerScreenshot();
            }, 1000 / this.screenshotFps);
            console.log(`Started viewer screenshot capture at ${this.screenshotFps} fps`);
        }, 3000);
    }

    async captureViewerScreenshot() {
        try {
            // Get the Neuroglancer viewer panel element
            const viewerPanel = document.querySelector('.ng-viewer-panel');
            if (!viewerPanel) {
                return;
            }

            // Use html2canvas to capture the rendered content
            // For now, let's try using the iframe's window directly
            if (!this.ngIframe || !this.ngIframe.contentWindow) {
                return;
            }

            // Try to call Neuroglancer's built-in screenshot function from the iframe
            try {
                const iframeWindow = this.ngIframe.contentWindow;
                const iframeDoc = this.ngIframe.contentDocument;
                
                // Debug: Check what's available
                console.log('[SCREENSHOT] iframe window keys:', Object.keys(iframeWindow).filter(k => k.toLowerCase().includes('viewer') || k.toLowerCase().includes('ng')));
                
                // Check the viewer object
                if (iframeWindow.viewer) {
                    console.log('[SCREENSHOT] viewer exists!', typeof iframeWindow.viewer);
                    console.log('[SCREENSHOT] viewer methods:', Object.keys(iframeWindow.viewer).filter(k => typeof iframeWindow.viewer[k] === 'function'));
                    console.log('[SCREENSHOT] has screenshot?', typeof iframeWindow.viewer.screenshot);
                }
                
                // Try multiple ways to access the viewer
                let viewer = iframeWindow.viewer || 
                             iframeWindow.ngviewer || 
                             iframeWindow.neuroglancer?.viewer ||
                             (iframeDoc && iframeDoc.viewer);
                
                if (!viewer) {
                    // Try to find it in the global scope
                    for (let key of Object.keys(iframeWindow)) {
                        if (iframeWindow[key] && typeof iframeWindow[key].screenshot === 'function') {
                            viewer = iframeWindow[key];
                            console.log(`[SCREENSHOT] Found viewer at window.${key}`);
                            break;
                        }
                    }
                }
                
                if (viewer && typeof viewer.screenshot === 'function') {
                    console.log('[SCREENSHOT] Calling Neuroglancer screenshot API from iframe');
                    const screenshot = await viewer.screenshot();
                    if (screenshot && screenshot.screenshot && screenshot.screenshot.image) {
                        // Convert ArrayBuffer/Uint8Array to base64
                        const imageData = screenshot.screenshot.image;
                        const base64 = this.arrayBufferToBase64(imageData);
                        await this.sendScreenshotToServer(base64);
                        return;
                    }
                } else {
                    console.log('[SCREENSHOT] No viewer object found in iframe');
                }
            } catch (e) {
                console.log('[SCREENSHOT] Neuroglancer API not available:', e.message);
            }

            // Fallback: capture using canvas (will be blank due to WebGL but we'll try)
            console.log('[SCREENSHOT] Falling back to canvas capture');
            
        } catch (e) {
            console.error('[SCREENSHOT] Failed to capture:', e);
        }
    }

    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return window.btoa(binary);
    }

    setupPageScreenshotCapture() {
        setTimeout(() => {
            this.screenshotInterval = setInterval(() => {
                this.capturePageScreenshot();
            }, 1000 / this.screenshotFps);
            console.log(`Started page screenshot capture at ${this.screenshotFps} fps`);
        }, 3000);
    }

    restartScreenshotCapture() {
        if (this.screenshotInterval) {
            clearInterval(this.screenshotInterval);
        }
        this.screenshotInterval = setInterval(() => {
            this.capturePageScreenshot();
        }, 1000 / this.screenshotFps);
        console.log(`Restarted screenshot capture at ${this.screenshotFps} fps`);
    }

    capturePageScreenshot() {
        try {
            // Create a canvas the size of the Neuroglancer viewer area
            const viewerContainer = document.querySelector('.ng-iframe-container');
            if (!viewerContainer) {
                return;
            }

            const rect = viewerContainer.getBoundingClientRect();
            const canvas = document.createElement('canvas');
            canvas.width = 800;
            canvas.height = 600;
            const ctx = canvas.getContext('2d');

            // Fill with a background
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Capture the canvas from inside the iframe
            if (this.ngIframe) {
                try {
                    const iframeDoc = this.ngIframe.contentDocument;
                    const iframeCanvas = iframeDoc.querySelector('canvas');
                    if (iframeCanvas) {
                        ctx.drawImage(iframeCanvas, 0, 0, canvas.width, canvas.height);
                    }
                } catch (e) {
                    console.log('[SCREENSHOT] Cannot access iframe canvas:', e.message);
                }
            }

            // Convert to JPEG
            const jpegDataUrl = canvas.toDataURL('image/jpeg', 0.8);
            const jpegBase64 = jpegDataUrl.split(',')[1];
            
            console.log(`[SCREENSHOT] Captured ${jpegBase64.length} chars`);
            this.sendScreenshotToServer(jpegBase64);

        } catch (e) {
            console.error('[SCREENSHOT] Page capture failed:', e);
        }
    }

    setupIframeScreenshotCapture() {
        // Wait for iframe to load, then start capturing screenshots from canvas
        setTimeout(() => {
            setInterval(() => {
                this.captureIframeCanvas();
            }, 1000 / this.screenshotFps);  // Capture at screenshotFps rate
            console.log(`Started iframe canvas capture at ${this.screenshotFps} fps`);
        }, 3000);  // Wait 3 seconds for Neuroglancer to load
    }

    captureIframeCanvas() {
        try {
            if (!this.ngIframe || !this.ngIframe.contentWindow) {
                return;
            }

            // Access the iframe's document (same-origin, so this works!)
            const iframeDoc = this.ngIframe.contentDocument || this.ngIframe.contentWindow.document;
            if (!iframeDoc) {
                return;
            }

            // Find ALL canvases and use the largest one (main render canvas)
            const canvases = iframeDoc.querySelectorAll('canvas');
            if (!canvases || canvases.length === 0) {
                console.log('[SCREENSHOT] No canvas found in iframe');
                return;
            }

            console.log(`[SCREENSHOT] Found ${canvases.length} canvases`);

            // Find the largest canvas (likely the main render canvas)
            let mainCanvas = null;
            let maxArea = 0;
            for (const canvas of canvases) {
                const area = canvas.width * canvas.height;
                console.log(`[SCREENSHOT] Canvas: ${canvas.width}x${canvas.height} (area=${area})`);
                if (area > maxArea) {
                    maxArea = area;
                    mainCanvas = canvas;
                }
            }

            if (!mainCanvas || mainCanvas.width === 0 || mainCanvas.height === 0) {
                console.log('[SCREENSHOT] No valid canvas found');
                return;
            }

            console.log(`[SCREENSHOT] Using canvas: ${mainCanvas.width}x${mainCanvas.height}`);

            // Try to read a pixel to check if canvas is tainted
            try {
                const testCtx = mainCanvas.getContext('2d') || mainCanvas.getContext('webgl') || mainCanvas.getContext('webgl2');
                if (testCtx) {
                    console.log(`[SCREENSHOT] Canvas context type: ${testCtx.constructor.name}`);
                }
            } catch (e) {
                console.log(`[SCREENSHOT] Canvas is tainted/cross-origin: ${e.message}`);
            }

            // Create a smaller canvas for efficiency
            const targetWidth = 800;
            const targetHeight = 600;
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = targetWidth;
            captureCanvas.height = targetHeight;
            const ctx = captureCanvas.getContext('2d');

            // Draw the Neuroglancer canvas onto our capture canvas (scaled down)
            ctx.drawImage(mainCanvas, 0, 0, targetWidth, targetHeight);

            // Convert to JPEG base64
            const jpegDataUrl = captureCanvas.toDataURL('image/jpeg', 0.8);
            const jpegBase64 = jpegDataUrl.split(',')[1];

            console.log(`[SCREENSHOT] Captured ${jpegBase64.length} bytes`);

            // Send to server
            this.sendScreenshotToServer(jpegBase64);

        } catch (e) {
            console.error('[SCREENSHOT] Failed to capture canvas:', e);
        }
    }

    async sendScreenshotToServer(jpegBase64) {
        try {
            const response = await fetch('/api/screenshot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    jpeg_b64: jpegBase64,
                    timestamp: Date.now()
                })
            });

            if (!response.ok) {
                console.error('[SCREENSHOT] Server rejected screenshot:', response.status);
            }
        } catch (e) {
            console.error('[SCREENSHOT] Failed to send screenshot to server:', e);
        }
    }

    startScreenshotCapture() {
        // Wait a bit for iframe to load
        setTimeout(() => {
            this.screenshotInterval = setInterval(() => {
                this.requestScreenshot();
            }, 1000 / this.screenshotFps);
            console.log(`Started screenshot capture at ${this.screenshotFps} fps`);
        }, 3000);
    }

    requestScreenshot() {
        if (!this.iframeReady || !this.ngIframe.contentWindow) {
            console.log('[SCREENSHOT] Iframe not ready');
            return;
        }

        const now = Date.now();
        // Throttle requests
        if (now - this.lastScreenshotTime < 1000 / this.screenshotFps) {
            return;
        }

        // Send message to iframe requesting a screenshot
        this.ngIframe.contentWindow.postMessage({
            type: 'captureScreenshot',
            width: 800,
            height: 600
        }, '*');

        this.lastScreenshotTime = now;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new NGLiveStream();
});
