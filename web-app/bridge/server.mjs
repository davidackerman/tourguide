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
import { WebSocketServer } from "ws";

const PORT = Number(process.env.TG_BRIDGE_PORT || 7723);
const VERSION = "0.1.0";
const OP_TIMEOUT_MS = Number(process.env.TG_BRIDGE_OP_TIMEOUT_MS || 30_000);

/** sessionId -> { record, ws } */
const sessions = new Map();
/** agent event-subscriber sockets */
const agents = new Set();
/** requestId -> { resolve, timer } */
const pending = new Map();

const now = () => new Date().toISOString();
const log = (...a) => console.log(`[bridge ${new Date().toLocaleTimeString()}]`, ...a);

// --- session selection (v0: most recently created running session) --------

function pickSession() {
  let best = null;
  for (const s of sessions.values()) {
    if (s.record.status !== "running") continue;
    if (!best || s.record.createdAt > best.record.createdAt) best = s;
  }
  return best;
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

function sendJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json", ...CORS });
  res.end(payload);
}

function relayOp(request) {
  return new Promise((resolve) => {
    const target = pickSession();
    if (!target) {
      resolve({ id: request.id, ok: false, error: { message: "no running Tourguide session" } });
      return;
    }
    const id = request.id || crypto.randomUUID();
    const timer = setTimeout(() => {
      pending.delete(id);
      resolve({ id, ok: false, error: { message: `op timed out after ${OP_TIMEOUT_MS}ms` } });
    }, OP_TIMEOUT_MS);
    pending.set(id, { resolve, timer });
    target.ws.send(JSON.stringify({ kind: "request", request: { ...request, id } }));
  });
}

const server = http.createServer((req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, CORS);
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
      const record = {
        sessionId,
        createdAt: existing?.record.createdAt ?? now(),
        lastSeenAt: now(),
        url: msg.session.url,
        mode: msg.session.mode,
        status: "running",
      };
      sessions.set(sessionId, { record, ws });
      ws.send(JSON.stringify({ kind: "registered" }));
      log(`browser session registered: ${sessionId} (${msg.session.mode})`);
      broadcastToAgents(connectionStatusEvent());
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

// Heartbeat: ping browsers, prune long-dead sessions.
setInterval(() => {
  for (const s of sessions.values()) {
    if (s.ws.readyState === s.ws.OPEN) s.ws.send(JSON.stringify({ kind: "ping" }));
  }
  broadcastToAgents({ type: "heartbeat", at: now() });
}, 20_000);

server.listen(PORT, () => {
  log(`Tourguide Workspace bridge v${VERSION} listening on http://localhost:${PORT}`);
  log(`  HTTP: GET /health, GET /sessions, POST /op`);
  log(`  WS:   /browser (session), /agent (events)`);
});
