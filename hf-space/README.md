---
title: Tourguide Analysis
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cloud analysis backend for the tourguide web-app.
---

# Tourguide Analysis Backend

This is the optional cloud-compute backend for the
[tourguide web-app](https://tourguide-8j4.pages.dev). The web-app normally
runs all analysis browser-side via Pyodide, but Pyodide is limited to
~4 GB of WASM memory. When a user's analysis is bigger than that they
can toggle "Run on backend" in the Custom Analysis modal, and the request
is forwarded here instead.

**Nothing is required on the user's side** — the web-app is already
configured to call this Space. If it's sleeping, a first request wakes
it (20–60 s cold start), after which it stays hot for 48 h.

## What runs here

Minimal FastAPI:
- `GET /api/health` — heartbeat + memory info + queue depth.
- `POST /api/analysis/run` — accepts a `CustomRequest` from the web-app,
  loads any referenced zarr layers, runs the user's Python code in a
  sandboxed subprocess, returns a `CustomResultMsg`.
- `WS /ws/bridge/<session_id>` — tunnel for local-folder zarrs: the
  browser pushes chunk bytes over this WS so the backend can process
  data that never left the user's laptop.
- `GET /api/data/<session_id>/<path>` — serves synthesized zarrs that
  analyses create (e.g. a contact-site mask) so Neuroglancer can load
  them as layers.

## Run your own instance

If you want isolated compute for heavy workloads, click **Duplicate this
Space** in the menu (top-right). HF will build your own copy under your
account. Copy the resulting URL (e.g.
`https://<yourname>-tourguide-analysis.hf.space`) and paste it into the
tourguide web-app at ⚙ AI → Analysis backend → URL. Everything else is
automatic.

## Security

Incoming Python code is checked against a whitelist of imports and a
deny-list of dangerous patterns (file writes outside `/tmp`, shell calls,
raw sockets, etc.). Per-request execution is in a subprocess with CPU
time + address-space rlimits and a wall-clock timeout. IP-based rate
limits (10 analysis requests / minute) are applied on top.
