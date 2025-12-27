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
        this.audioEnabled = false;  // Track if user has enabled audio
        this.audioQueue = [];  // Queue audio until enabled
        this.isPlayingAudio = false;  // Track if audio is currently playing

        // Screenshot capture settings
        this.screenshotInterval = null;
        this.screenshotFps = 0.5;  // Default 0.5 fps (1 screenshot every 2 seconds)
        this.lastScreenshotTime = 0;
        this.iframeReady = false;

        // Recording state
        this.isRecording = false;
        this.currentSessionId = null;
        this.recordedFrameCount = 0;

        // Query mode state
        this.currentMode = 'query';  // Default to query mode
        this.voiceEnabled = true;  // Voice narration enabled by default
        this.chatHistory = [];

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
        this.enableAudioBtn = document.getElementById('enable-audio-btn');

        // Recording UI elements
        this.transitionTypeSelect = document.getElementById('transition-type');
        this.startRecordingBtn = document.getElementById('start-recording');
        this.stopRecordingBtn = document.getElementById('stop-recording');
        this.createMovieBtn = document.getElementById('create-movie');
        this.recordingStatusText = document.getElementById('recording-status-text');
        this.recordingFrameCount = document.getElementById('recording-frame-count');
        this.recFrameCountSpan = document.getElementById('rec-frame-count');
        this.compilationProgress = document.getElementById('compilation-progress');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');

        // Query mode UI elements
        this.modeExploreBtn = document.getElementById('mode-explore');
        this.modeQueryBtn = document.getElementById('mode-query');
        this.voiceToggleBtn = document.getElementById('voice-toggle');
        this.chatPanel = document.getElementById('chat-panel');
        this.chatMessagesContainer = document.getElementById('chat-messages');
        this.verboseMessagesContainer = document.getElementById('verbose-messages');
        this.chatInput = document.getElementById('chat-input');
        this.chatSend = document.getElementById('chat-send');
        this.sidePanels = document.querySelector('.side-panels');

        // Chat tab elements
        this.chatTabBtns = document.querySelectorAll('.chat-tab-btn');
        this.chatTabContents = document.querySelectorAll('.chat-tab-content');

        // Enable audio on ANY user interaction
        const enableAudioOnInteraction = () => {
            if (!this.audioEnabled) {
                this.audioEnabled = true;
                if (this.enableAudioBtn) {
                    this.enableAudioBtn.style.display = 'none';
                }
                console.log('[AUDIO] Audio enabled by user interaction');
                // Play any queued audio
                if (this.audioQueue.length > 0 && !this.isPlayingAudio) {
                    this.playNextAudio();
                }
            }
        };

        // Listen for any interaction
        ['click', 'keydown', 'touchstart'].forEach(event => {
            document.addEventListener(event, enableAudioOnInteraction, { once: true });
        });

        // Setup audio enable button as backup
        if (this.enableAudioBtn) {
            this.enableAudioBtn.addEventListener('click', enableAudioOnInteraction);
        }

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

        // Setup recording controls
        this.setupRecordingControls();

        // Setup query mode controls
        this.setupModeToggle();
        this.setupVoiceToggle();
        this.setupChatHandlers();

        this.loadNeuroglancerURL();
        this.connect();
        this.syncMode();
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
        } else if (data.type === 'mode_change') {
            this.currentMode = data.mode;
            this.updateModeUI();
            console.log(`[MODE] Received mode change: ${data.mode}`);
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
        console.log('Audio data present:', !!data.audio);
        if (data.audio) {
            console.log('Audio data length:', data.audio.length);
        }

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

        // Queue audio if available and voice is enabled
        if (data.audio && this.voiceEnabled) {
            console.log('Queueing audio for playback');
            this.queueAudio(data.audio);
        } else if (data.audio && !this.voiceEnabled) {
            console.log('Audio available but voice is disabled');
        } else {
            console.warn('No audio data in narration message');
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
        
        // Start playing if not already playing and audio is enabled
        if (!this.isPlayingAudio && this.audioEnabled) {
            this.playNextAudio();
        } else if (!this.audioEnabled) {
            console.log('[AUDIO] Audio queued, waiting for user to enable audio');
            // Show the enable audio button if it's hidden
            if (this.enableAudioBtn) {
                this.enableAudioBtn.style.display = 'block';
            }
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
            
            // Auto-detect audio format by checking file signature
            // WAV: starts with "RIFF", MP3: starts with "ID3" or 0xFF
            let mimeType = 'audio/mpeg'; // default to MP3
            if (bytes.length >= 4) {
                if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
                    mimeType = 'audio/wav'; // RIFF header = WAV
                }
            }
            
            // Create blob and URL
            const blob = new Blob([bytes], { type: mimeType });
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
            // Skip screenshot capture in query mode (screenshots only needed for AI narration in explore mode)
            if (this.currentMode === 'query') {
                return;
            }

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

    // Recording methods
    setupRecordingControls() {
        if (!this.startRecordingBtn || !this.stopRecordingBtn) {
            console.warn('[RECORDING] Recording controls not found');
            return;
        }

        // Start recording
        this.startRecordingBtn.addEventListener('click', async () => {
            await this.startRecording();
        });

        // Stop recording
        this.stopRecordingBtn.addEventListener('click', async () => {
            await this.stopRecording();
        });

        // Create movie
        if (this.createMovieBtn) {
            this.createMovieBtn.addEventListener('click', async () => {
                await this.createMovie();
            });
        }

        console.log('[RECORDING] Controls initialized');
    }

    async startRecording() {
        try {
            const response = await fetch('/api/recording/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fps: this.screenshotFps,
                    transition_type: this.transitionTypeSelect.value,
                    transition_duration: 0.5
                })
            });

            const data = await response.json();
            if (data.status === 'ok') {
                this.isRecording = true;
                this.currentSessionId = data.session_id;
                this.recordedFrameCount = 0;

                // Update UI
                this.startRecordingBtn.disabled = true;
                this.stopRecordingBtn.disabled = false;
                this.recordingStatusText.textContent = 'Recording...';
                this.recordingStatusText.style.color = '#dc3545';
                this.recordingFrameCount.style.display = 'inline';
                this.transitionTypeSelect.disabled = true;
                this.createMovieBtn.style.display = 'none';

                // Start polling recording status
                this.startRecordingStatusPolling();

                console.log('[RECORDING] Started:', data.session_id);
            } else {
                alert('Failed to start recording: ' + data.message);
            }
        } catch (e) {
            console.error('[RECORDING] Start error:', e);
            alert('Failed to start recording');
        }
    }

    async stopRecording() {
        try {
            const response = await fetch('/api/recording/stop', {
                method: 'POST'
            });

            const data = await response.json();
            if (data.status === 'ok') {
                this.isRecording = false;

                // Stop polling
                this.stopRecordingStatusPolling();

                // Update UI
                this.startRecordingBtn.disabled = false;
                this.stopRecordingBtn.disabled = true;
                this.recordingStatusText.textContent = `Stopped (${data.frame_count} frames)`;
                this.recordingStatusText.style.color = '#28a745';
                this.createMovieBtn.style.display = 'inline-block';
                this.transitionTypeSelect.disabled = false;

                console.log('[RECORDING] Stopped:', data.frame_count, 'frames');
            } else {
                alert('Failed to stop recording: ' + data.message);
            }
        } catch (e) {
            console.error('[RECORDING] Stop error:', e);
            alert('Failed to stop recording');
        }
    }

    async createMovie() {
        try {
            this.createMovieBtn.disabled = true;
            this.compilationProgress.style.display = 'block';

            // Get current transition type from dropdown
            const transitionType = this.transitionTypeSelect.value;

            const response = await fetch('/api/recording/compile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    transition_type: transitionType
                })
            });

            const data = await response.json();
            if (data.status === 'ok') {
                console.log('[RECORDING] Compilation started:', data.session_id, 'with', transitionType);

                // Start polling compilation status
                this.startCompilationStatusPolling(data.session_id);
            } else {
                alert('Failed to compile movie: ' + data.message);
                this.compilationProgress.style.display = 'none';
                this.createMovieBtn.disabled = false;
            }
        } catch (e) {
            console.error('[RECORDING] Compile error:', e);
            alert('Failed to compile movie');
            this.compilationProgress.style.display = 'none';
            this.createMovieBtn.disabled = false;
        }
    }

    startRecordingStatusPolling() {
        this.recordingStatusInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/recording/status');
                const data = await response.json();
                if (data.is_recording) {
                    this.recFrameCountSpan.textContent = data.frame_count;
                }
            } catch (e) {
                console.error('[RECORDING] Status poll error:', e);
            }
        }, 1000);
    }

    stopRecordingStatusPolling() {
        if (this.recordingStatusInterval) {
            clearInterval(this.recordingStatusInterval);
            this.recordingStatusInterval = null;
        }
    }

    startCompilationStatusPolling(sessionId) {
        this.compilationStatusInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/recording/compile/status/${sessionId}`);
                const data = await response.json();

                if (data.status === 'completed') {
                    // Success!
                    clearInterval(this.compilationStatusInterval);
                    this.compilationProgress.style.display = 'none';
                    this.recordingStatusText.textContent = 'Movie ready!';
                    this.recordingStatusText.style.color = '#007bff';
                    this.createMovieBtn.style.display = 'none';

                    alert('Movie compilation completed! Check the recordings directory.');
                } else if (data.status === 'error') {
                    clearInterval(this.compilationStatusInterval);
                    this.compilationProgress.style.display = 'none';
                    this.createMovieBtn.disabled = false;
                    alert('Movie compilation failed');
                }
                // Spinner is already visible while polling, no need to update anything
            } catch (e) {
                console.error('[RECORDING] Compilation status poll error:', e);
            }
        }, 2000);
    }

    // Query Mode Methods
    async syncMode() {
        try {
            const response = await fetch('/api/mode');
            const data = await response.json();
            this.currentMode = data.mode;
            this.updateModeUI();
            console.log(`[MODE] Synced to ${data.mode} mode`);
            if (data.mode === 'query') {
                console.log('[MODE] Screenshot capture disabled in query mode');
            } else {
                console.log('[MODE] Screenshot capture enabled in explore mode');
            }
        } catch (e) {
            console.error('[MODE] Failed to sync mode:', e);
        }
    }

    setupModeToggle() {
        if (!this.modeExploreBtn || !this.modeQueryBtn) return;

        this.modeExploreBtn.addEventListener('click', () => {
            this.switchMode('explore');
        });

        this.modeQueryBtn.addEventListener('click', () => {
            this.switchMode('query');
        });
    }

    setupVoiceToggle() {
        if (!this.voiceToggleBtn) return;

        this.voiceToggleBtn.addEventListener('click', () => {
            this.voiceEnabled = !this.voiceEnabled;
            this.updateVoiceUI();
            console.log(`[VOICE] Voice narration ${this.voiceEnabled ? 'enabled' : 'disabled'}`);
        });
    }

    updateVoiceUI() {
        if (!this.voiceToggleBtn) return;

        if (this.voiceEnabled) {
            this.voiceToggleBtn.classList.add('active');
            this.voiceToggleBtn.querySelector('.voice-icon').textContent = '🔊';
        } else {
            this.voiceToggleBtn.classList.remove('active');
            this.voiceToggleBtn.querySelector('.voice-icon').textContent = '🔇';
        }
    }

    async switchMode(mode) {
        try {
            const response = await fetch('/api/mode/set', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });

            const data = await response.json();
            if (data.status === 'ok') {
                this.currentMode = mode;
                this.updateModeUI();
                console.log(`[MODE] Switched to ${mode} mode`);
                if (mode === 'query') {
                    console.log('[MODE] Screenshot capture disabled in query mode');
                } else {
                    console.log('[MODE] Screenshot capture enabled in explore mode');
                }
            } else {
                console.error('[MODE] Failed to switch mode:', data.message);
            }
        } catch (e) {
            console.error('[MODE] Failed to switch mode:', e);
        }
    }

    updateModeUI() {
        if (this.currentMode === 'explore') {
            // Show explore panels
            if (this.sidePanels) this.sidePanels.style.display = 'flex';
            if (this.chatPanel) this.chatPanel.style.display = 'none';

            // Update button states
            if (this.modeExploreBtn) this.modeExploreBtn.classList.add('active');
            if (this.modeQueryBtn) this.modeQueryBtn.classList.remove('active');
        } else if (this.currentMode === 'query') {
            // Show chat panel
            if (this.sidePanels) this.sidePanels.style.display = 'none';
            if (this.chatPanel) this.chatPanel.style.display = 'block';

            // Update button states
            if (this.modeExploreBtn) this.modeExploreBtn.classList.remove('active');
            if (this.modeQueryBtn) this.modeQueryBtn.classList.add('active');

            // Focus on input
            if (this.chatInput) this.chatInput.focus();
        }
    }

    setupChatHandlers() {
        if (!this.chatSend || !this.chatInput) return;

        this.chatSend.addEventListener('click', () => {
            this.sendQuery();
        });

        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendQuery();
            }
        });

        // Setup tab switching
        this.chatTabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabName = btn.dataset.tab;
                this.switchChatTab(tabName);
            });
        });
    }

    switchChatTab(tabName) {
        // Update button states
        this.chatTabBtns.forEach(btn => {
            if (btn.dataset.tab === tabName) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update tab content visibility
        this.chatTabContents.forEach(content => {
            if (content.id === `chat-tab-${tabName}`) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
    }

    async sendQuery() {
        const query = this.chatInput.value.trim();
        if (!query) return;

        // Add user message to chat
        this.addChatMessage('user', query);
        this.chatInput.value = '';

        // Show loading indicator
        const loadingId = this.addChatMessage('loading', 'Processing...');

        try {
            const response = await fetch('/api/query/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    generate_audio: this.voiceEnabled
                })
            });

            const data = await response.json();

            // Remove loading indicator
            this.removeChatMessage(loadingId);

            if (data.status === 'ok') {
                const result = data.result;

                // Add AI response to chat
                this.addChatMessage('ai', result.answer || 'Query processed');

                // Add verbose log entry
                this.addVerboseLogEntry(query, result);

                // Play audio if available and voice is enabled
                if (data.audio && this.voiceEnabled) {
                    this.queueAudio(data.audio);
                } else if (data.audio && !this.voiceEnabled) {
                    console.log('[QUERY] Audio available but voice is disabled');
                }

                // Log navigation if applicable
                if (result.type === 'navigation') {
                    console.log('[QUERY] Navigated to:', result.navigation);
                }
            } else {
                this.addChatMessage('error', data.message || 'Query failed');
            }
        } catch (e) {
            this.removeChatMessage(loadingId);
            this.addChatMessage('error', 'Failed to send query: ' + e.message);
            console.error('[QUERY] Error:', e);
        }
    }

    addChatMessage(type, text) {
        if (!this.chatMessagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message chat-${type}`;
        messageDiv.textContent = text;
        messageDiv.dataset.id = Date.now();

        // Remove placeholder if exists
        const placeholder = this.chatMessagesContainer.querySelector('.chat-placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        this.chatMessagesContainer.appendChild(messageDiv);
        this.chatMessagesContainer.scrollTop = this.chatMessagesContainer.scrollHeight;

        return messageDiv.dataset.id;
    }

    removeChatMessage(id) {
        if (!this.chatMessagesContainer) return;

        const message = this.chatMessagesContainer.querySelector(`[data-id="${id}"]`);
        if (message) {
            message.remove();
        }
    }

    addVerboseLogEntry(query, result) {
        if (!this.verboseMessagesContainer) return;

        const entryDiv = document.createElement('div');
        entryDiv.className = 'verbose-entry';  // Start collapsed by default

        // Create header (always visible) with query and toggle arrow
        let html = `
            <div class="verbose-entry-header">
                <div class="verbose-query">❯ ${this.escapeHtml(query)}</div>
                <div class="verbose-toggle">▶</div>
            </div>
            <div class="verbose-entry-content">`;

        // Model info
        if (result.model) {
            html += `<div class="verbose-section">
                <div class="verbose-section-title">AI Model</div>
                <div><strong>${result.model}</strong></div>
            </div>`;
        }

        // AI Interactions (full prompts and responses)
        if (result.ai_interactions && result.ai_interactions.length > 0) {
            html += `<div class="verbose-section">
                <div class="verbose-section-title">AI Interactions (${result.ai_interactions.length})</div>`;

            result.ai_interactions.forEach((interaction, idx) => {
                const typeLabel = this.formatInteractionType(interaction.type);
                html += `
                    <div class="ai-interaction">
                        <div class="ai-interaction-header">
                            <strong>${idx + 1}. ${typeLabel}</strong>
                            ${interaction.model ? `<span class="ai-model-badge">${interaction.model}</span>` : ''}
                        </div>
                        <div class="ai-interaction-prompt">
                            <div class="ai-interaction-label">Prompt:</div>
                            <pre>${this.escapeHtml(interaction.prompt)}</pre>
                        </div>
                        <div class="ai-interaction-response">
                            <div class="ai-interaction-label">Response:</div>
                            <pre>${this.escapeHtml(interaction.response)}</pre>
                        </div>
                        ${interaction.cleaned_sql ? `
                            <div class="ai-interaction-cleaned">
                                <div class="ai-interaction-label">Cleaned SQL:</div>
                                <pre>${this.escapeHtml(interaction.cleaned_sql)}</pre>
                            </div>
                        ` : ''}
                        ${interaction.retry ? `
                            <div class="ai-interaction-retry">
                                ⚠️ Retry attempt (previous error: ${this.escapeHtml(interaction.previous_error || 'unknown')})
                            </div>
                        ` : ''}
                    </div>
                `;
            });

            html += `</div>`;
        }

        // Query type
        if (result.type) {
            html += `<div class="verbose-section">
                <div class="verbose-section-title">Query Type</div>
                <div>${result.type}</div>
            </div>`;
        }

        // SQL query
        if (result.sql) {
            html += `<div class="verbose-section">
                <div class="verbose-section-title">Generated SQL</div>
                <div class="verbose-sql">${this.escapeHtml(result.sql)}</div>
            </div>`;
        }

        // Results
        if (result.results && result.results.length > 0) {
            const resultsPreview = JSON.stringify(result.results, null, 2);
            html += `<div class="verbose-section">
                <div class="verbose-section-title">Query Results (${result.results.length} row${result.results.length !== 1 ? 's' : ''})</div>
                <div class="verbose-result">${this.escapeHtml(resultsPreview)}</div>
            </div>`;
        }

        // Navigation command
        if (result.navigation) {
            const navPreview = JSON.stringify(result.navigation, null, 2);
            html += `<div class="verbose-section">
                <div class="verbose-section-title">Navigation Command</div>
                <div class="verbose-result">${this.escapeHtml(navPreview)}</div>
            </div>`;
        }

        // Visualization command
        if (result.visualization) {
            const vizPreview = JSON.stringify(result.visualization, null, 2);
            html += `<div class="verbose-section">
                <div class="verbose-section-title">Visualization Command</div>
                <div class="verbose-result">${this.escapeHtml(vizPreview)}</div>
            </div>`;
        }

        // Answer
        if (result.answer) {
            html += `<div class="verbose-section">
                <div class="verbose-section-title">Answer</div>
                <div>${this.escapeHtml(result.answer)}</div>
            </div>`;
        }

        // Timing info
        if (result.timing) {
            const timingInfo = Object.entries(result.timing)
                .map(([key, value]) => `${key}: ${value.toFixed(3)}s`)
                .join(', ');
            html += `<div class="verbose-timing">⏱ ${timingInfo}</div>`;
        }

        // Close the content wrapper
        html += `</div>`;

        entryDiv.innerHTML = html;

        // Add click handler to toggle expansion
        const header = entryDiv.querySelector('.verbose-entry-header');
        header.addEventListener('click', () => {
            entryDiv.classList.toggle('expanded');
        });

        // Remove placeholder if exists
        const placeholder = this.verboseMessagesContainer.querySelector('.chat-placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        this.verboseMessagesContainer.appendChild(entryDiv);
        this.verboseMessagesContainer.scrollTop = this.verboseMessagesContainer.scrollHeight;
    }

    formatInteractionType(type) {
        const typeMap = {
            'intent_classification': '🎯 Intent Classification',
            'sql_generation': '💾 SQL Generation',
            'answer_formatting': '✍️ Answer Formatting',
            'visualization_answer': '🎨 Visualization Answer'
        };
        return typeMap[type] || type;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new NGLiveStream();
});
