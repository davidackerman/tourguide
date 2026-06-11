# Tourguide Workspace bridge

A browser tab can't host a server, but the live viewer / DB / plots all live
in the browser. This small Node process is the hub that lets external agents
(MCP adapter, Python SDK, local scripts) drive a running Tourguide session.

```
 external agent ──HTTP /op──┐                  ┌──WS /browser── browser
 (MCP / SDK / curl)         ├──►  bridge  ◄─────┤   (?mode=workspace)
 external agent ──WS /agent─┘     (relay)       └── registers a session
```

- **HTTP** carries request/response ops: `GET /health`, `GET /sessions`,
  `POST /op` (body = `WorkspaceRequest`, returns `WorkspaceResponse`).
- **WebSocket** carries live streams: `/agent` (action history + connection
  status events to agents) and `/browser` (relayed op-requests to the tab).

The browser is the source of truth for workspace state; the bridge keeps
only lightweight session metadata and picks the most recently created
running session for v0.

## Run

```bash
cd web-app
npm install
npm run bridge          # starts on http://localhost:7723
npm run dev             # Vite; open http://localhost:5173/?mode=workspace
```

The workspace tab connects to `ws://localhost:7723/browser` automatically.
Override the port with `?bridgePort=NNNN` on the page URL and
`TG_BRIDGE_PORT=NNNN` on the server.

## Try it from the terminal

```bash
node bridge/test_client.mjs health
node bridge/test_client.mjs sessions
node bridge/test_client.mjs watch                                  # stream events
node bridge/test_client.mjs op get_session
node bridge/test_client.mjs op fly_to '{"position":[1000,2000,3000]}'
node bridge/test_client.mjs op select_segments '{"layer":"mito","segmentIds":["12","34"]}'
```

`op` requires a workspace tab to be open (it relays to the browser); the
others work with the bridge alone.

## Environment

| Var | Default | Meaning |
| --- | --- | --- |
| `TG_BRIDGE_PORT` | `7723` | HTTP + WS port |
| `TG_BRIDGE_OP_TIMEOUT_MS` | `30000` | how long `/op` waits for the browser |
