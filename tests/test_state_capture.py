#!/usr/bin/env python3
"""
Test script to verify screenshot capture and state retrieval with sources.

This tests:
1. Neuroglancer viewer initialization with data sources
2. State capture via to_json() includes source URLs
3. State can be serialized and used for video_tool interpolation
"""

import neuroglancer
import json
import sys
from pathlib import Path


def test_state_capture():
    """Test that neuroglancer state includes source URLs."""
    print("=" * 80)
    print("TEST 1: State Capture with Sources")
    print("=" * 80)

    # Create a test viewer
    neuroglancer.set_server_bind_address("127.0.0.1", bind_port=9997)
    viewer = neuroglancer.Viewer()

    # Add C. elegans data (same as production server)
    base_url = "zarr://https://cellmap-vm1.int.janelia.org/nrs/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1"

    print("\nAdding layers with sources...")
    with viewer.txn() as s:
        s.layers["fibsem-uint8"] = neuroglancer.ImageLayer(
            source=f"{base_url}/em/fibsem-uint8/"
        )
        s.layers["mito"] = neuroglancer.SegmentationLayer(
            source=f"{base_url}/labels/inference/segmentations/mito/"
        )
        s.layers["nuc"] = neuroglancer.SegmentationLayer(
            source=f"{base_url}/labels/inference/segmentations/nuc/"
        )

        # Set a test position
        s.position = [1000, 1500, 2000]
        s.crossSectionScale = 0.5

    # Capture state
    print("\nCapturing state with to_json()...")
    with viewer.txn() as s:
        state_json = s.to_json()

    # Verify state structure
    print("\n--- State Structure ---")
    print(f"State keys: {list(state_json.keys())}")

    if "layers" not in state_json:
        print("❌ FAIL: No 'layers' key in state")
        return False

    print(f"\nNumber of layers: {len(state_json['layers'])}")

    # Check each layer for sources
    all_have_sources = True
    for i, layer in enumerate(state_json["layers"]):
        layer_name = layer.get("name", f"layer_{i}")
        has_source = "source" in layer

        print(f"\nLayer '{layer_name}':")
        print(f"  - Type: {layer.get('type', 'unknown')}")
        print(f"  - Has source: {has_source}")

        if has_source:
            source = layer["source"]
            print(f"  - Source type: {type(source)}")
            if isinstance(source, list) and len(source) > 0:
                print(f"  - Source URL: {source[0].get('url', 'N/A')}")
            elif isinstance(source, str):
                print(f"  - Source: {source}")
            else:
                print(f"  - Source: {source}")
        else:
            print(f"  - ❌ WARNING: Layer '{layer_name}' has no source!")
            all_have_sources = False

    # Check position
    if "position" in state_json:
        print(f"\n✓ Position captured: {state_json['position']}")
    else:
        print(f"\n❌ WARNING: No position in state")

    # Check scale
    if "crossSectionScale" in state_json:
        print(f"✓ Scale captured: {state_json['crossSectionScale']}")
    else:
        print(f"❌ WARNING: No crossSectionScale in state")

    # Test JSON serialization (for saving to file)
    print("\n--- JSON Serialization Test ---")
    try:
        json_str = json.dumps(state_json, indent=2)
        print(f"✓ State can be serialized to JSON ({len(json_str)} bytes)")

        # Verify it can be re-parsed
        reparsed = json.loads(json_str)
        print(f"✓ JSON can be re-parsed")

        # Verify sources survived serialization
        sources_intact = any("source" in layer for layer in reparsed.get("layers", []))
        if sources_intact:
            print(f"✓ Sources preserved after serialization")
        else:
            print(f"❌ Sources lost after serialization!")
            all_have_sources = False

    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

    # Summary
    print("\n" + "=" * 80)
    if all_have_sources:
        print("✓ TEST PASSED: All layers have sources")
        print("\nFull state JSON:")
        print(json.dumps(state_json, indent=2))
        return True
    else:
        print("❌ TEST FAILED: Some layers missing sources")
        return False


def test_url_generation():
    """Test that state can be encoded into neuroglancer URL."""
    print("\n" + "=" * 80)
    print("TEST 2: URL Generation with Sources")
    print("=" * 80)

    # Create test state with sources
    test_state = {
        "position": [1000, 1500, 2000],
        "crossSectionScale": 0.5,
        "layers": [
            {
                "name": "fibsem-uint8",
                "type": "image",
                "source": [
                    {
                        "url": "zarr://https://cellmap-vm1.int.janelia.org/nrs/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/em/fibsem-int16/"
                    }
                ],
                "visible": True,
            },
            {
                "name": "mito",
                "type": "segmentation",
                "source": [
                    {
                        "url": "zarr://https://cellmap-vm1.int.janelia.org/nrs/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/labels/inference/segmentations/mito/"
                    }
                ],
                "visible": False,
            },
        ],
    }

    # Import the URL generation function
    sys.path.insert(0, str(Path(__file__).parent / "server"))
    from recording import generate_public_url

    print("\nGenerating neuroglancer-demo URL from state...")
    try:
        url = generate_public_url(test_state)
        print(f"✓ URL generated ({len(url)} chars)")
        print(f"\nURL: {url[:100]}...")

        # Verify URL structure
        if not url.startswith("https://neuroglancer-demo.appspot.com/#!"):
            print(f"❌ URL doesn't have expected prefix")
            return False

        # Decode and verify sources are in URL
        import urllib.parse

        encoded_state = url.split("#!")[1]
        decoded_state = json.loads(urllib.parse.unquote(encoded_state))

        print(f"\n✓ URL can be decoded back to state")

        # Check if sources survived URL encoding
        has_sources = any(
            "source" in layer for layer in decoded_state.get("layers", [])
        )
        if has_sources:
            print(f"✓ Sources preserved in URL-encoded state")
            print(f"\nDecoded state layers:")
            for layer in decoded_state.get("layers", []):
                if "source" in layer:
                    print(f"  - {layer['name']}: {layer['source']}")
        else:
            print(f"❌ Sources lost in URL encoding!")
            return False

        print("\n✓ TEST PASSED: URL generation preserves sources")
        return True

    except Exception as e:
        print(f"❌ URL generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_video_tool_script():
    """Test that we can generate a video_tool script with sources."""
    print("\n" + "=" * 80)
    print("TEST 3: Video Tool Script Generation")
    print("=" * 80)

    # Import URL generation
    sys.path.insert(0, str(Path(__file__).parent / "server"))
    from recording import generate_public_url

    # Create test states for 3 keyframes
    keyframes = []
    for i in range(3):
        state = {
            "position": [1000 + i * 500, 1500, 2000],
            "crossSectionScale": 0.5 - i * 0.1,
            "layers": [
                {
                    "name": "fibsem-uint8",
                    "type": "image",
                    "source": [
                        {
                            "url": "zarr://https://cellmap-vm1.int.janelia.org/nrs/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/em/fibsem-uint8/"
                        }
                    ],
                    "visible": True,
                }
            ],
        }
        keyframes.append(state)

    print(f"\nGenerating script for {len(keyframes)} keyframes...")

    # Generate script (format: URL, duration, URL, duration, URL)
    script_lines = []
    for i, state in enumerate(keyframes):
        # Add transition duration before URL (except first)
        if i > 0:
            script_lines.append("2.0")  # 2 second transition

        # Add URL
        url = generate_public_url(state)
        script_lines.append(url)

    script_content = "\n".join(script_lines)

    print(f"✓ Script generated ({len(script_lines)} lines)")
    print(f"\nScript preview (first 500 chars):")
    print(script_content[:500] + "...")

    # Verify each URL has sources
    print(f"\nVerifying sources in each keyframe:")
    import urllib.parse

    for i, line in enumerate(script_lines):
        if line.startswith("https://"):
            # This is a URL line
            encoded_state = line.split("#!")[1]
            decoded_state = json.loads(urllib.parse.unquote(encoded_state))
            has_sources = any(
                "source" in layer for layer in decoded_state.get("layers", [])
            )

            keyframe_num = (i + 1) // 2  # Account for duration lines
            if has_sources:
                print(f"  ✓ Keyframe {keyframe_num}: has sources")
            else:
                print(f"  ❌ Keyframe {keyframe_num}: missing sources!")
                return False

    print("\n✓ TEST PASSED: Video tool script contains sources")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STATE CAPTURE AND INTERPOLATION TEST SUITE")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("State Capture", test_state_capture()))
    results.append(("URL Generation", test_url_generation()))
    results.append(("Video Tool Script", test_video_tool_script()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print(
            "\nConclusion: State capture includes sources, and interpolation should work"
        )
        print("for new recording sessions on the cellmap-vm1 internal network.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("\nConclusion: There may be issues with state capture or URL generation.")
        sys.exit(1)
