// Tourguide Workspace API — local bridge server.
//
// A browser tab can't host a server, but the live viewer/DB/plots live in
// the browser. So this small Node process is the hub:
//
//   external agent ──HTTP /op──┐                  ┌──WS /browser── browser
//   (MCP / SDK / curl)         ├──► bridge ◄──────┤   (workspace mode)
//   external agent ──WS /agent─┘   (relay)        └── registers a session
//
// HTTP carries request/response ops (/op, /health, /sessions); WebSocket
// carries the live event stream to agents (/agent) and relayed op-requests
// to the browser (/browser). The browser is the source of truth for
// workspace state; the bridge keeps only lightweight session metadata.
//
// Run:  TG_BRIDGE_PORT=7723 node bridge/server.mjs
// Deps: ws (devDependency). No other runtime deps.

import http from "node:http";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { WebSocketServer } from "ws";
import {
  saveState,
  listStates,
  getState,
  stateDir,
  saveShareState,
  getShareState,
  saveSessionState,
  getSessionState,
  isValidShareId,
} from "./state_store.mjs";

const PORT = Number(process.env.TG_BRIDGE_PORT || 7723);
const VERSION = "0.1.0";
const OP_TIMEOUT_MS = Number(process.env.TG_BRIDGE_OP_TIMEOUT_MS || 30_000);

/** sessionId -> { record, ws } */
const sessions = new Map();
/** agent event-subscriber sockets */
const agents = new Set();
/** requestId -> { resolve, timer } */
const pending = new Map();
/** monotonic counter for human-readable tab labels (workspace-1, -2, …) */
let labelCounter = 0;
/** latest viewer-fly target (for the embedded Python viewer control channel) */
let lastFly = null;
let flySeq = 0;

const now = () => new Date().toISOString();
const log = (...a) => console.log(`[bridge ${new Date().toLocaleTimeString()}]`, ...a);

// --- session selection (v0: most recently created running session) --------

function isLive(s) {
  // Running AND ponged within the staleness window: a tab that has gone away
  // but isn't pruned yet would hit a dead socket if we routed to it.
  return s.record.status === "running" && Date.now() - Date.parse(s.record.lastSeenAt) <= STALE_MS;
}

function liveSessions() {
  return [...sessions.values()].filter(isLive);
}

function pickSession() {
  let best = null;
  for (const s of liveSessions()) {
    if (!best || s.record.createdAt > best.record.createdAt) best = s;
  }
  return best;
}

// Choose the target tab for a relayed op. With an explicit sessionId we route
// there (and only there). Without one we route to the sole live tab — but if
// several are open we refuse to guess (that silent guess was the original
// "drove the wrong/blank tab" bug); the caller must say which.
function resolveTarget(sessionId) {
  if (sessionId) {
    const s = sessions.get(sessionId);
    if (!s || !isLive(s)) return { error: `workspace tab '${sessionId}' is not connected` };
    return { session: s };
  }
  const live = liveSessions();
  if (live.length === 0) return { error: "no running Tourguide session" };
  if (live.length === 1) return { session: live[0] };
  const labels = live.map((s) => `${s.record.label} (${s.record.sessionId})`).join(", ");
  return {
    error:
      `multiple workspace tabs are open: ${labels}. ` +
      "Specify which to drive (pass `session`), or open a dedicated one.",
  };
}

function sessionRecords() {
  return [...sessions.values()].map((s) => s.record);
}

function broadcastToAgents(event) {
  const msg = JSON.stringify(event);
  for (const ws of agents) {
    if (ws.readyState === ws.OPEN) ws.send(msg);
  }
}

function connectionStatusEvent() {
  const s = pickSession();
  return {
    type: "connection_status",
    status: s ? "connected" : "disconnected",
    sessionId: s?.record.sessionId,
  };
}

// --- HTTP server -----------------------------------------------------------

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "content-type",
};

// Agent-computed artifacts (e.g. a zarr the agent wrote) are served here so the
// viewer can load them as layers. The bridge binds to all interfaces, so the
// URL is http://localhost:PORT/artifacts/… for you and http://<machine-ip>:PORT
// /artifacts/… for others on the same LAN/VPN. Default dir ~/.tourguide/
// artifacts (override with TG_ARTIFACTS_DIR). Returns real 404s (the vite SPA
// fallback returned index.html, which broke NG's zarr probing) and CORP so the
// cross-origin-isolated page can fetch it.
const ARTIFACTS_DIR = (process.env.TG_ARTIFACTS_DIR || "").trim() ||
  path.join(os.homedir(), ".tourguide", "artifacts");

function serveArtifact(req, res, pathname) {
  const rel = decodeURIComponent(pathname.slice("/artifacts/".length));
  const base = path.resolve(ARTIFACTS_DIR);
  const full = path.resolve(base, rel);
  if (full !== base && !full.startsWith(base + path.sep)) {
    res.writeHead(403, CORS);
    res.end();
    return;
  }
  fs.readFile(full, (err, data) => {
    if (err) {
      res.writeHead(404, CORS);
      res.end("not found");
      return;
    }
    const ct = /\.(zattrs|zgroup|zarray|json)$/.test(full)
      ? "application/json"
      : "application/octet-stream";
    res.writeHead(200, {
      ...CORS,
      "Cross-Origin-Resource-Policy": "cross-origin",
      "content-type": ct,
      "cache-control": "no-cache",
    });
    res.end(data);
  });
}

function sendJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json", ...CORS });
  res.end(payload);
}

function relayOp(request) {
  return new Promise((resolve) => {
    const resolved = resolveTarget(request.session);
    if (resolved.error) {
      resolve({ id: request.id, ok: false, error: { message: resolved.error } });
      return;
    }
    const target = resolved.session;
    const id = request.id || crypto.randomUUID();
    const timer = setTimeout(() => {
      pending.delete(id);
      resolve({ id, ok: false, error: { message: `op timed out after ${OP_TIMEOUT_MS}ms` } });
    }, OP_TIMEOUT_MS);
    pending.set(id, { resolve, timer });
    target.ws.send(JSON.stringify({ kind: "request", request: { ...request, id } }));
  });
}

// --- disk-backed saved states ----------------------------------------------
//
// Disk is the durable source of truth (the browser's localStorage is a
// per-tab cache for its own panel). save → ask the browser to serialize the
// full state, then write it; list/get → read disk; restore → read disk, hand
// the full state to the browser to apply (works even in a fresh tab whose
// localStorage is empty).

const STATE_OPS = new Set(["save_session_state", "list_saved_states", "restore_session_state"]);

async function handleStateOp(request) {
  try {
    if (request.op === "list_saved_states") {
      return { id: request.id, ok: true, result: listStates() };
    }
    if (request.op === "save_session_state") {
      // The browser owns the live viewer, so it serializes the full state;
      // the bridge persists it and returns a trimmed result + its path.
      const relayed = await relayOp(request);
      if (!relayed.ok) return relayed;
      const record = relayed.result;
      if (!record || typeof record !== "object" || !record.id) {
        return { id: request.id, ok: false, error: { message: "browser returned no serializable state to save" } };
      }
      return { id: request.id, ok: true, result: saveState(record) };
    }
    if (request.op === "restore_session_state") {
      const wanted = request.params?.id;
      const record = wanted ? getState(wanted) : null;
      // Found on disk → apply it directly (pass the full state inline). Not on
      // disk → fall through to the browser's own localStorage lookup by id.
      const relayRequest = record
        ? { ...request, params: { ...request.params, state: record } }
        : request;
      return await relayOp(relayRequest);
    }
  } catch (err) {
    return { id: request.id, ok: false, error: { message: `state op failed: ${err.message}` } };
  }
  return { id: request.id, ok: false, error: { message: `unhandled state op: ${request.op}` } };
}

const server = http.createServer((req, res) => {
  if (req.method === "OPTIONS") {
    // Private Network Access: a public HTTPS page (e.g. a hosted Tourguide
    // build) reaching this localhost bridge triggers a PNA preflight. Echo
    // the allow header so Chrome lets the real request through. localhost is
    // already mixed-content-exempt; PNA is the extra gate Chrome adds for
    // public→private requests. (The page still connects out to us; we never
    // reach into it — same trust model as Babylon NME's local MCP server.)
    const pna = req.headers["access-control-request-private-network"] === "true"
      ? { "Access-Control-Allow-Private-Network": "true" }
      : {};
    res.writeHead(204, { ...CORS, ...pna });
    res.end();
    return;
  }
  const url = new URL(req.url, `http://localhost:${PORT}`);

  if (req.method === "GET" && url.pathname === "/health") {
    sendJson(res, 200, { ok: true, version: VERSION, sessions: sessions.size });
    return;
  }

  if (req.method === "GET" && url.pathname === "/sessions") {
    sendJson(res, 200, sessionRecords());
    return;
  }

  // Serve agent-computed artifacts (zarr, meshes, …) for the viewer to load as
  // layers — reachable on the LAN/VPN via the machine IP, not just localhost.
  if (req.method === "GET" && url.pathname.startsWith("/artifacts/")) {
    serveArtifact(req, res, url.pathname);
    return;
  }

  // Share-state store for short LAN Tourguide links (…?state=<id>). POST a
  // viewer state, get an id; the recipient's browser GETs it back over the LAN.
  if (req.method === "POST" && url.pathname === "/share-state") {
    let raw = "";
    req.on("data", (c) => {
      raw += c;
      if (raw.length > 16 * 1024 * 1024) req.destroy();
    });
    req.on("end", () => {
      let state;
      try {
        state = JSON.parse(raw);
      } catch {
        sendJson(res, 400, { ok: false, error: { message: "invalid JSON" } });
        return;
      }
      const id = crypto.randomUUID().slice(0, 12);
      try {
        saveShareState(id, state);
        sendJson(res, 200, { ok: true, id });
      } catch (err) {
        sendJson(res, 500, { ok: false, error: { message: err.message } });
      }
    });
    return;
  }
  // Viewer-fly control channel for the embedded Python NG viewer. The browser
  // (click-to-fly) and the agent POST a target; the viewer-holder polls GET and
  // applies it. Decouples "who wants to fly" from "who owns the viewer".
  if (req.method === "POST" && url.pathname === "/viewer-fly") {
    let raw = "";
    req.on("data", (c) => { raw += c; if (raw.length > 64 * 1024) req.destroy(); });
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        lastFly = { seq: ++flySeq, ...body };
        sendJson(res, 200, { ok: true, seq: flySeq });
      } catch {
        sendJson(res, 400, { ok: false, error: { message: "invalid JSON" } });
      }
    });
    return;
  }
  if (req.method === "GET" && url.pathname === "/viewer-fly") {
    sendJson(res, 200, lastFly || { seq: 0 });
    return;
  }
  if (req.method === "GET" && url.pathname.startsWith("/share-state/")) {
    const id = url.pathname.slice("/share-state/".length);
    const state = isValidShareId(id) ? getShareState(id) : null;
    if (state == null) {
      sendJson(res, 404, { ok: false, error: { message: "share state not found" } });
    } else {
      sendJson(res, 200, state);
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/op") {
    let raw = "";
    req.on("data", (c) => {
      raw += c;
      if (raw.length > 64 * 1024 * 1024) req.destroy(); // 64MB guard
    });
    req.on("end", async () => {
      let request;
      try {
        request = JSON.parse(raw);
      } catch {
        sendJson(res, 400, { ok: false, error: { message: "invalid JSON" } });
        return;
      }
      // launch_or_attach is answered by the bridge itself: report the chosen
      // session if one is running, else signal the launcher to start one.
      if (request.op === "launch_or_attach") {
        const s = pickSession();
        if (s) sendJson(res, 200, { id: request.id, ok: true, result: s.record });
        else sendJson(res, 200, { id: request.id, ok: false, error: { message: "no running session" } });
        return;
      }
      // Saved states persist to disk here (the browser sandbox can't), so the
      // bridge intercepts the three state ops instead of leaving them
      // browser/localStorage-only.
      if (STATE_OPS.has(request.op)) {
        sendJson(res, 200, await handleStateOp(request));
        return;
      }
      const response = await relayOp(request);
      sendJson(res, 200, response);
    });
    return;
  }

  sendJson(res, 404, { ok: false, error: { message: `not found: ${url.pathname}` } });
});

// --- WebSocket server (routes by path) -------------------------------------

const wss = new WebSocketServer({ server });

wss.on("connection", (ws, req) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname === "/agent") {
    handleAgent(ws);
  } else if (url.pathname === "/browser") {
    handleBrowser(ws);
  } else {
    ws.close(1008, "unknown path");
  }
});

function handleAgent(ws) {
  agents.add(ws);
  log(`agent subscriber connected (${agents.size} total)`);
  // Greet with current connection status so the agent knows immediately
  // whether a session is live.
  ws.send(JSON.stringify(connectionStatusEvent()));
  ws.on("close", () => {
    agents.delete(ws);
    log(`agent subscriber disconnected (${agents.size} total)`);
  });
  ws.on("message", () => {
    /* agents are event subscribers; ops go over HTTP. Ignore inbound. */
  });
}

function handleBrowser(ws) {
  let sessionId = null;
  ws.on("message", (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.kind === "register" && msg.session) {
      sessionId = msg.session.sessionId;
      const existing = sessions.get(sessionId);
      // A label is assigned once per tab and reused across reconnects, so a
      // tab keeps a stable human-readable name (for the "which tab?" choice
      // and the browser title) even when the bridge restarts.
      const label = existing?.record.label ?? `workspace-${++labelCounter}`;
      const record = {
        sessionId,
        label,
        createdAt: existing?.record.createdAt ?? now(),
        lastSeenAt: now(),
        url: msg.session.url,
        mode: msg.session.mode,
        status: "running",
      };
      sessions.set(sessionId, { record, ws });
      ws.send(JSON.stringify({ kind: "registered", label }));
      log(`browser session registered: ${label} ${sessionId} (${msg.session.mode})`);
      broadcastToAgents(connectionStatusEvent());
      // Reopening a ?session=<id> link should restore its workspace: if we
      // have a persisted snapshot for this id, send it back for the page to
      // apply (the page may be a fresh tab whose localStorage is empty).
      const restored = getSessionState(sessionId);
      if (restored) {
        ws.send(JSON.stringify({ kind: "restore", state: restored }));
        log(`sent restore snapshot to ${label} ${sessionId}`);
      }
    } else if (msg.kind === "persist" && msg.state) {
      // Rolling auto-save of the page's current workspace snapshot, keyed by
      // the session id, so the next open of this link restores it.
      if (sessionId) saveSessionState(sessionId, msg.state);
    } else if (msg.kind === "response" && msg.response) {
      const p = pending.get(msg.response.id);
      if (p) {
        clearTimeout(p.timer);
        pending.delete(msg.response.id);
        p.resolve(msg.response);
      }
    } else if (msg.kind === "event" && msg.event) {
      broadcastToAgents(msg.event);
    } else if (msg.kind === "pong") {
      const s = sessionId && sessions.get(sessionId);
      if (s) s.record.lastSeenAt = now();
    }
  });
  ws.on("close", () => {
    if (sessionId && sessions.get(sessionId)?.ws === ws) {
      const s = sessions.get(sessionId);
      s.record.status = "disconnected";
      log(`browser session disconnected: ${sessionId}`);
      broadcastToAgents(connectionStatusEvent());
    }
  });
}

// Heartbeat: ping browsers, and actually prune dead sessions so a new
// thread never attaches to a tab that has quietly gone away.
//
//  - A tab that closes cleanly fires ws "close" and is marked disconnected
//    immediately (see handleBrowser). But a tab that dies WITHOUT a clean
//    close — laptop sleep, killed browser, half-open TCP — never fires it,
//    so without this it would read "running" forever and `pickSession`
//    would hand it to the next agent. We catch those by staleness: a live
//    tab pongs every interval, so no pong for STALE_MS means it's gone.
//  - Disconnected records are dropped after DROP_MS so they don't pile up
//    (we had 16 stale sessions accumulate before this existed).
const PING_INTERVAL_MS = 20_000;
const STALE_MS = Number(process.env.TG_BRIDGE_STALE_MS || 70_000); // ~3 missed pongs
const DROP_MS = Number(process.env.TG_BRIDGE_DROP_MS || 300_000); // forget after 5 min

setInterval(() => {
  const nowMs = Date.now();
  let changed = false;
  for (const [id, s] of sessions) {
    if (s.ws.readyState === s.ws.OPEN) s.ws.send(JSON.stringify({ kind: "ping" }));
    const idleMs = nowMs - Date.parse(s.record.lastSeenAt);
    if (s.record.status === "running" && idleMs > STALE_MS) {
      s.record.status = "disconnected";
      changed = true;
      log(`session pruned (no pong for ${Math.round(idleMs / 1000)}s): ${id}`);
      try {
        s.ws.terminate();
      } catch {
        /* already gone */
      }
    }
    if (s.record.status === "disconnected" && idleMs > DROP_MS) {
      sessions.delete(id);
    }
  }
  if (changed) broadcastToAgents(connectionStatusEvent());
  broadcastToAgents({ type: "heartbeat", at: now() });
}, PING_INTERVAL_MS);

server.listen(PORT, () => {
  log(`Tourguide Workspace bridge v${VERSION} listening on http://localhost:${PORT}`);
  log(`  HTTP: GET /health, GET /sessions, POST /op`);
  log(`  WS:   /browser (session), /agent (events)`);
});
