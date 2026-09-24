# Tourguide Workspace bridge

A browser tab can't host a server, but the live viewer / DB / plots all live
in the browser. This small Node process is the hub that lets external agents
(MCP adapter, Python SDK, local scripts) drive a running Tourguide session.

```
 external agent ──HTTP /op──┐                  ┌──WS /browser── browser
 (MCP / SDK / curl)         ├──►  bridge  ◄─────┤   (?mode=workspace)
 external agent ──/events───┘     (relay)       └── registers a session
```

- **HTTP**: `GET /health` (open), `GET /sessions`, `GET /events?since=&wait=&types=`,
  `POST /op` (body = `WorkspaceRequest`, returns `WorkspaceResponse`),
  `POST /share-state` → id and `GET /share-state/<id>` (short share links),
  `GET /artifacts/*` (agent-computed zarr / meshes from `~/.tourguide/artifacts`,
  Range-capable), `/viewer-fly` (embedded Python viewer control channel).
- **WebSocket**: `/agent` (live events to agents) and `/browser` (relayed
  op-requests to the tab, rolling `persist` snapshots, `restore` on reconnect).

The browser is the source of truth for workspace state; the bridge keeps
session metadata, disk-backed saved states / per-session snapshots / share
blobs under `~/.tourguide/`, and a 500-event ring buffer. Ops carry an
optional `session` id; without one the bridge routes to the sole live tab and
refuses to guess among several. Read-only viewers (`?view=1`) are never
targets and never persist.

## Security

- Binds `127.0.0.1` only (`TG_BRIDGE_HOST` to widen).
- Bearer token required on `/op`, `/sessions`, `/events`, `POST /share-state`,
  `/viewer-fly` and both WS paths: `Authorization: Bearer …`,
  `x-tourguide-token`, or `?token=` (WS). From `TG_BRIDGE_TOKEN`, else
  generated; written 0600 as JSON `{token, viewToken}` to
  `<tmpdir>/tourguide-bridge-<port>.token`. `TG_BRIDGE_NO_AUTH=1` disables.
  Tokens are never printed.
- The **view token** only lets a tab register as a read-only viewer; share
  links carry it.
- `GET /health`, `GET /artifacts/*` and `GET /share-state/<id>` are open
  (Neuroglancer can't attach headers); they serve derived data by unguessable
  id and are only reachable off-machine when LAN-bound.
- Requests with an `Origin` outside loopback / this machine's addresses /
  `TG_BRIDGE_ALLOWED_ORIGINS` (default: the hosted Tourguide page) are refused
  (403); CORS echoes allowed origins only.
- 64 MB body limit (413).

## Run

```bash
cd web-app && npm install
npm run workspace:preview   # bridge + built app together; prints the tokened URL
# or separately:
npm run bridge              # 127.0.0.1:7723
npm run preview             # then open http://localhost:5173/?mode=workspace&bridgeToken=<token>
```

Override the port with `?bridgePort=NNNN` on the page URL and
`TG_BRIDGE_PORT=NNNN` on the server.

## Try it from the terminal

```bash
node bridge/test_client.mjs health
node bridge/test_client.mjs sessions
node bridge/test_client.mjs watch                                  # stream events
node bridge/test_client.mjs events 0                               # poll events
node bridge/test_client.mjs op get_session
node bridge/test_client.mjs op fly_to '{"position":[1000,2000,3000]}'
node bridge/test_client.mjs op select_segments '{"layer":"mito","segmentIds":["12","34"]}'
node bridge/test_client.mjs screenshot view.png
```

The CLI reads the token from the token file (or `TG_BRIDGE_TOKEN`).

## Environment

| Var | Default | Meaning |
| --- | --- | --- |
| `TG_BRIDGE_PORT` | `7723` | HTTP + WS port |
| `TG_BRIDGE_HOST` | `127.0.0.1` | bind address |
| `TG_BRIDGE_TOKEN` | _(generated)_ | bearer token |
| `TG_BRIDGE_NO_AUTH` | unset | `1` disables auth |
| `TG_BRIDGE_OP_TIMEOUT_MS` | `30000` | how long `/op` waits for the browser |
| `TG_BRIDGE_ALLOWED_ORIGINS` | hosted page | comma-separated extra browser origins |
| `TG_ARTIFACTS_DIR` | `~/.tourguide/artifacts` | agent-computed layers served at `/artifacts/` |
| `TG_STATE_DIR` / `TG_SESSION_STATE_DIR` / `TG_SHARE_STATE_DIR` | `~/.tourguide/…` | saved states / session snapshots / share blobs |
| `TG_BRIDGE_STALE_MS` / `TG_BRIDGE_DROP_MS` | `70000` / `300000` | session liveness pruning |
