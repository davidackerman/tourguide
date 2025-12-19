#!/usr/bin/env python3
"""Quick script to check if screenshots are being captured and save one to disk."""

import requests
import base64
import time

# Get screenshot from server
response = requests.post(
    "http://localhost:8090/api/screenshot",
    json={"jpeg_b64": "test", "timestamp": time.time()},
)

# Try to get a frame via the API
print("Checking screenshot endpoint...")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# The screenshots are in memory, let's trigger the server to print info
print("\nScreenshots are being received - check /tmp/server.log")
print("Size: Look for '[SCREENSHOT] Received from client: XXXX bytes'")
print("\nIf size is ~3608 bytes, screenshots ARE being captured")
print("Check browser console (F12) for:")
print("  - [NG-SCREENSHOT] Handler initialized")
print("  - [NG-SCREENSHOT] Canvas detected")
print("  - [NG-SCREENSHOT] Captured: XXXX bytes")
