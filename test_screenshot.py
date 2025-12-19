#!/usr/bin/env python3
"""
Minimal test to see if screenshot API works at all.
"""
import neuroglancer
import time

# Create viewer
neuroglancer.set_server_bind_address("0.0.0.0", bind_port=9999)
viewer = neuroglancer.Viewer()

# Add simple data
with viewer.txn() as s:
    s.layers["em"] = neuroglancer.ImageLayer(
        source="precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe"
    )
    s.position = [119132, 80036, 3840]

print(f"Neuroglancer URL: {viewer}")
print("Open this URL in your browser and wait 5 seconds...")
time.sleep(5)

print("\nAttempting screenshot with 5 second timeout...")
import threading

result = {"done": False, "data": None}


def try_screenshot():
    try:
        s = viewer.screenshot()
        result["data"] = s
        result["done"] = True
        print(f"SUCCESS! Screenshot size: {len(s.screenshot.image)} bytes")
    except Exception as e:
        print(f"FAILED: {e}")
        result["done"] = True


t = threading.Thread(target=try_screenshot)
t.start()
t.join(timeout=15.0)

if not result["done"]:
    print("TIMEOUT: screenshot() never returned after 15 seconds")
else:
    if result["data"]:
        print(
            f"Screenshot successful: {result['data'].screenshot.width}x{result['data'].screenshot.height}"
        )
    else:
        print("Screenshot failed")

print("\nKeeping server alive for 30 seconds...")
time.sleep(30)
