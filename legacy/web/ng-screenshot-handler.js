/**
 * Neuroglancer Screenshot Handler
 * This script runs inside the Neuroglancer iframe and handles screenshot requests
 * from the parent window via postMessage.
 */

(function() {
    console.log('[NG-SCREENSHOT] Handler initialized');

    // Wait for Neuroglancer to load
    function waitForNeuroglancer(callback) {
        const checkInterval = setInterval(() => {
            // Check if Neuroglancer's viewer exists in the global scope
            if (window.viewer || window.ngviewer || document.querySelector('canvas')) {
                clearInterval(checkInterval);
                console.log('[NG-SCREENSHOT] Neuroglancer detected');
                callback();
            }
        }, 500);

        // Timeout after 30 seconds
        setTimeout(() => {
            clearInterval(checkInterval);
            console.error('[NG-SCREENSHOT] Neuroglancer not found within timeout');
        }, 30000);
    }

    // Capture screenshot from Neuroglancer canvas
    function captureScreenshot(width, height) {
        try {
            // Find the Neuroglancer canvas element
            const canvas = document.querySelector('canvas.neuroglancer-gl-canvas') ||
                          document.querySelector('canvas.webgl-canvas') ||
                          document.querySelector('canvas') ||
                          document.querySelector('.neuroglancer-viewer canvas');

            if (!canvas) {
                console.error('[NG-SCREENSHOT] No canvas found');
                return null;
            }

            console.log('[NG-SCREENSHOT] Found canvas:', canvas.width, 'x', canvas.height);

            // For WebGL canvases, we need to capture directly without intermediate canvas
            // because toDataURL may return blank due to WebGL security
            try {
                // Try direct capture first
                const jpeg_b64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                
                // Check if we got a valid image (not just black)
                if (jpeg_b64 && jpeg_b64.length > 1000) {
                    console.log('[NG-SCREENSHOT] Direct capture:', jpeg_b64.length, 'bytes');
                    return jpeg_b64;
                } else {
                    console.warn('[NG-SCREENSHOT] Direct capture returned small/empty image, trying alternative...');
                }
            } catch (e) {
                console.warn('[NG-SCREENSHOT] Direct capture failed:', e.message);
            }

            // Alternative: Try reading WebGL pixels directly
            try {
                const gl = canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                if (gl) {
                    console.log('[NG-SCREENSHOT] Using WebGL pixel readback');
                    
                    // Create temp canvas for pixel data
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = width || canvas.width;
                    tempCanvas.height = height || canvas.height;
                    const ctx = tempCanvas.getContext('2d');
                    
                    // Read pixels from WebGL
                    const pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
                    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
                    
                    // Create ImageData and flip vertically (WebGL is upside down)
                    const imageData = ctx.createImageData(tempCanvas.width, tempCanvas.height);
                    for (let y = 0; y < tempCanvas.height; y++) {
                        for (let x = 0; x < tempCanvas.width; x++) {
                            const srcY = tempCanvas.height - y - 1;  // Flip vertically
                            const srcIdx = (srcY * tempCanvas.width + x) * 4;
                            const dstIdx = (y * tempCanvas.width + x) * 4;
                            imageData.data[dstIdx] = pixels[srcIdx];
                            imageData.data[dstIdx + 1] = pixels[srcIdx + 1];
                            imageData.data[dstIdx + 2] = pixels[srcIdx + 2];
                            imageData.data[dstIdx + 3] = pixels[srcIdx + 3];
                        }
                    }
                    
                    ctx.putImageData(imageData, 0, 0);
                    const jpeg_b64 = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                    console.log('[NG-SCREENSHOT] WebGL readPixels:', jpeg_b64.length, 'bytes');
                    return jpeg_b64;
                }
            } catch (e) {
                console.error('[NG-SCREENSHOT] WebGL pixel readback failed:', e);
            }

            // Fallback: standard canvas copy
            const tempCanvas = document.createElement('canvas');
            const ctx = tempCanvas.getContext('2d');
            tempCanvas.width = width || canvas.width;
            tempCanvas.height = height || canvas.height;
            ctx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);
            const jpeg_b64 = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
            console.log('[NG-SCREENSHOT] Captured screenshot:', jpeg_b64.length, 'bytes (base64)');
            return jpeg_b64;

        } catch (error) {
            console.error('[NG-SCREENSHOT] Error capturing screenshot:', error);
            return null;
        }
    }

    // Listen for screenshot requests from parent
    window.addEventListener('message', (event) => {
        // In production, validate event.origin for security
        // if (event.origin !== 'https://expected-origin.com') return;

        if (event.data && event.data.type === 'captureScreenshot') {
            console.log('[NG-SCREENSHOT] Received capture request');

            const width = event.data.width || 800;
            const height = event.data.height || 600;

            const jpeg_b64 = captureScreenshot(width, height);

            if (jpeg_b64) {
                // Send screenshot back to parent
                window.parent.postMessage({
                    type: 'screenshot',
                    jpeg_b64: jpeg_b64,
                    timestamp: Date.now()
                }, '*');

                console.log('[NG-SCREENSHOT] Sent screenshot to parent');
            } else {
                console.error('[NG-SCREENSHOT] Failed to capture screenshot');
            }
        }
    });

    // Notify parent when ready
    waitForNeuroglancer(() => {
        window.parent.postMessage({
            type: 'ready'
        }, '*');
        console.log('[NG-SCREENSHOT] Notified parent that iframe is ready');
    });

})();
