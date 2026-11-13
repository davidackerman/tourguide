// Tourguide Workspace API — local bridge server.
//
// A browser tab can't host a server, but the live viewer/DB/plots live in
// the browser. So this small Node process is the hub:
//
//   external agent ──HTTP /op──┐                  ┌──WS /browser── browser
//   (MCP / SDK / curl)         ├──► bridge ◄──────┤   (workspace mode)
//   external agent ──/events───┘   (relay)        └── registers a session
//
// HTTP carries request/response ops (/op, /health, /sessions, /events) plus
// artifact + share-state serving; WebSocket carries the live event stream to
// agents (/agent) and relayed op-requests to the browser (/browser). The
// browser is the source of truth for workspace state; the bridge keeps
// session metadata, disk-backed saved states, and a short event ring buffer.
//
// Security posture — this process can read every table in the workspace and
// push arbitrary layers into the viewer, so it is locked down by default:
//   * binds 127.0.0.1                      TG_BRIDGE_HOST=0.0.0.0 to share on the LAN
//   * bearer token on everything that reads workspace state or drives the
//     viewer (/op, /sessions, /events, POST /share-state, /viewer-fly, both WS
//     paths). TG_BRIDGE_TOKEN, else generated; written 0600 as JSON
//     {token, viewToken} to <tmpdir>/tourguide-bridge-<port>.token so local
//     clients find it. TG_BRIDGE_NO_AUTH=1 disables (single-user machine only).
//   * a second, weaker VIEW token that only lets a tab register as a read-only
//     viewer (?view=1) — share links carry this one, never the full token.
//   * Origin allowlist: loopback, this machine's own addresses, and
//     TG_BRIDGE_ALLOWED_ORIGINS (default includes the hosted Tourguide page).
//     A random web page open in the same browser can't drive the workspace.
//   * CORS echoes allowed origins only (no wildcard).
//   * GET /artifacts/* and GET /share-state/<id> are served WITHOUT a token
//     (Neuroglancer fetches artifact chunks itself and cannot add headers) —
//     they only expose agent-computed derived data by unguessable id, and only
//     off-machine when you opt into a LAN bind.
//
// Run:  node bridge/server.mjs            (TG_BRIDGE_PORT=7723 by default)
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
  saveShareState,
  getShareState,
  saveSessionState,
  getSessionState,
  isValidShareId,
} from "./state_store.mjs";

const PORT = Number(process.env.TG_BRIDGE_PORT || 7723);
const HOST = process.env.TG_BRIDGE_HOST || "127.0.0.1";
const VERSION = "0.3.0";
const OP_TIMEOUT_MS = Number(process.env.TG_BRIDGE_OP_TIMEOUT_MS || 30_000);
const MAX_BODY_BYTES = 64 * 1024 * 1024;
const EVENT_BUFFER = 500;
const NO_AUTH = process.env.TG_BRIDGE_NO_AUTH === "1";

const now = () => new Date().toISOString();
const log = (...a) => console.log(`[bridge ${new Date().toLocaleTimeString()}]`, ...a);

// --- auth ------------------------------------------------------------------

export const tokenFilePath = (port = PORT) => path.join(os.tmpdir(), `tourguide-bridge-${port}.token`);

function initTokens() {
  if (NO_AUTH) return { token: null, viewToken: null };
  const token = process.env.TG_BRIDGE_TOKEN || crypto.randomBytes(24).toString("base64url");
  const viewToken = process.env.TG_BRIDGE_VIEW_TOKEN || crypto.randomBytes(18).toString("base64url");
  try {
    fs.writeFileSync(tokenFilePath(), JSON.stringify({ token, viewToken }), { mode: 0o600 });
    fs.chmodSync(tokenFilePath(), 0o600); // writeFileSync honors mode only on create
  } catch (err) {
    log(`warning: could not write token file ${tokenFilePath()}: ${err.message}`);
  }
  return { token, viewToken };
}

const { token: TOKEN, viewToken: VIEW_TOKEN } = initTokens();

function presentedToken(req, url) {
  const auth = req.headers["authorization"];
  if (typeof auth === "string" && auth.toLowerCase().startsWith("bearer ")) return auth.slice(7).trim();
  const hdr = req.headers["x-tourguide-token"];
  if (typeof hdr === "string" && hdr) return hdr;
  return url.searchParams.get("token") || "";
}

function safeEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  return crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b));
}

/** "full" | "view" | null — which credential the request presented. */
function authLevel(req, url) {
  if (!TOKEN) return "full";
  const given = presentedToken(req, url);
  if (safeEqual(given, TOKEN)) return "full";
  if (safeEqual(given, VIEW_TOKEN)) return "view";
  return null;
}

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

function machineAddresses() {
  const out = new Set();
  try {
    for (const ifaces of Object.values(os.networkInterfaces())) {
      for (const i of ifaces || []) out.add(i.family === "IPv6" ? `[${i.address}]` : i.address);
    }
  } catch {
    /* ignore */
  }
  out.add(os.hostname());
  return out;
}

const EXTRA_ORIGINS = new Set(
  (process.env.TG_BRIDGE_ALLOWED_ORIGINS ?? "https://tourguide-8j4.pages.dev")
    .split(",")
    .map((s) => s.trim().replace(/\/+$/, ""))
    .filter(Boolean),
);

/** True when the request carries no Origin (non-browser client), a loopback
 *  Origin, an Origin on one of this machine's own addresses (LAN-served page),
 *  or an explicitly allowed hosted origin. */
function originOk(req) {
  const origin = req.headers["origin"];
  if (!origin) return true;
  try {
    const u = new URL(origin);
    if (LOOPBACK_HOSTS.has(u.hostname)) return true;
    if (machineAddresses().has(u.hostname.replace(/^\[|\]$/g, "")) || machineAddresses().has(u.hostname)) return true;
    return EXTRA_ORIGINS.has(origin.replace(/\/+$/, ""));
  } catch {
    return false;
  }
}

function corsHeaders(req) {
  const origin = req.headers["origin"];
  if (!origin || !originOk(req)) return {};
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    // `range` so Neuroglancer's mesh/volume range requests survive the CORS
    // preflight; expose Content-Range/Accept-Ranges so its fetcher can plan them.
    "Access-Control-Allow-Headers": "content-type, range, authorization, x-tourguide-token",
    "Access-Control-Expose-Headers": "content-length, content-range, accept-ranges",
    Vary: "Origin",
  };
}

// --- state -----------------------------------------------------------------

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
/** ring buffer of recent events for HTTP polling agents */
const events = [];
let eventSeq = 0;
/** long-poll waiters on /events */
const eventWaiters = new Set();

// --- session selection ------------------------------------------------------

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
// several are open we refuse to guess; the caller must say which.
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

function publishEvent(event) {
  const entry = { seq: ++eventSeq, at: now(), ...event };
  events.push(entry);
  if (events.length > EVENT_BUFFER) events.splice(0, events.length - EVENT_BUFFER);
  const msg = JSON.stringify(entry);
  for (const ws of agents) {
    if (ws.readyState === ws.OPEN) ws.send(msg);
  }
  for (const w of eventWaiters) w();
}

function connectionStatusEvent() {
  const s = pickSession();
  return {
    type: "connection_status",
    status: s ? "connected" : "disconnected",
    sessionId: s?.record.sessionId,
  };
}

// --- artifacts ---------------------------------------------------------------
//
// Agent-computed artifacts (e.g. a zarr or meshes the agent wrote) are served
// here so the viewer can load them as layers: http://<bridge host>:PORT/artifacts/…
// Default dir ~/.tourguide/artifacts (TG_ARTIFACTS_DIR). Real 404s (the vite
// SPA fallback returned index.html, which broke NG's zarr probing) and CORP so
// the cross-origin-isolated page can fetch it. Served without a token because
// Neuroglancer's fetcher cannot attach one; path-traversal-safe.
const ARTIFACTS_DIR = (process.env.TG_ARTIFACTS_DIR || "").trim() ||
  path.join(os.homedir(), ".tourguide", "artifacts");

function serveArtifact(req, res, pathname) {
  const rel = decodeURIComponent(pathname.slice("/artifacts/".length));
  const base = path.resolve(ARTIFACTS_DIR);
  const full = path.resolve(base, rel);
  if (full !== base && !full.startsWith(base + path.sep)) {
    res.writeHead(403, corsHeaders(req));
    res.end();
    return;
  }
  fs.stat(full, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404, corsHeaders(req));
      res.end("not found");
      return;
    }
    const ct = /\.(zattrs|zgroup|zarray|json)$/.test(full)
      ? "application/json"
      : "application/octet-stream";
    const headers = {
      ...corsHeaders(req),
      "Cross-Origin-Resource-Policy": "cross-origin",
      "content-type": ct,
      "cache-control": "no-cache",
      "accept-ranges": "bytes",
    };
    // "bytes=START-END" (END or START may be empty; "bytes=-N" = last N bytes).
    const m = /^bytes=(\d*)-(\d*)$/.exec((req.headers["range"] || "").trim());
    if (m && (m[1] !== "" || m[2] !== "")) {
      let start = m[1] !== "" ? parseInt(m[1], 10) : st.size - parseInt(m[2], 10);
      let end = m[1] !== "" && m[2] !== "" ? parseInt(m[2], 10) : st.size - 1;
      start = Math.max(0, start);
      end = Math.min(end, st.size - 1);
      if (Number.isNaN(start) || Number.isNaN(end) || start > end || start >= st.size) {
        res.writeHead(416, { ...headers, "content-range": `bytes */${st.size}` });
        res.end();
        return;
      }
      res.writeHead(206, {
        ...headers,
        "content-range": `bytes ${start}-${end}/${st.size}`,
        "content-length": String(end - start + 1),
      });
      fs.createReadStream(full, { start, end }).pipe(res);
      return;
    }
    res.writeHead(200, { ...headers, "content-length": String(st.size) });
    fs.createReadStream(full).pipe(res);
  });
}

// --- HTTP helpers -----------------------------------------------------------

function sendJson(req, res, status, body) {
  res.writeHead(status, { "content-type": "application/json", ...corsHeaders(req) });
  res.end(JSON.stringify(body));
}

function readBody(req, limit = MAX_BODY_BYTES) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error(`request body exceeds ${limit} bytes`));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function readJson(req, res, limit) {
  let raw;
  try {
    raw = await readBody(req, limit);
  } catch (err) {
    sendJson(req, res, 413, { ok: false, error: { message: err.message } });
    return undefined;
  }
  try {
    return JSON.parse(raw);
  } catch {
    sendJson(req, res, 400, { ok: false, error: { message: "invalid JSON" } });
    return undefined;
  }
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

// --- HTTP server -------------------------------------------------------------

function unauthorized(req, res) {
  sendJson(req, res, 401, {
    ok: false,
    error: {
      message:
        "unauthorized: pass the bridge token (Authorization: Bearer …). " +
        `It is in TG_BRIDGE_TOKEN or ${tokenFilePath()}.`,
    },
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${HOST}:${PORT}`);

  if (req.method === "OPTIONS") {
    // Private Network Access: a public HTTPS page (the hosted Tourguide build)
    // reaching this localhost bridge triggers a PNA preflight. Echo the allow
    // header so Chrome lets the real request through.
    const pna = req.headers["access-control-request-private-network"] === "true"
      ? { "Access-Control-Allow-Private-Network": "true" }
      : {};
    res.writeHead(originOk(req) ? 204 : 403, { ...corsHeaders(req), ...pna });
    res.end();
    return;
  }
  if (!originOk(req)) {
    sendJson(req, res, 403, { ok: false, error: { message: "forbidden origin" } });
    return;
  }

  // --- open endpoints ---------------------------------------------------------
  if (req.method === "GET" && url.pathname === "/health") {
    sendJson(req, res, 200, { ok: true, version: VERSION, sessions: sessions.size, auth: !!TOKEN, host: HOST });
    return;
  }
  if (req.method === "GET" && url.pathname.startsWith("/artifacts/")) {
    serveArtifact(req, res, url.pathname);
    return;
  }
  if (req.method === "GET" && url.pathname.startsWith("/share-state/")) {
    const id = url.pathname.slice("/share-state/".length);
    const state = isValidShareId(id) ? getShareState(id) : null;
    if (state == null) sendJson(req, res, 404, { ok: false, error: { message: "share state not found" } });
    else sendJson(req, res, 200, state);
    return;
  }

  // --- token-gated endpoints --------------------------------------------------
  const level = authLevel(req, url);
  if (level !== "full") {
    unauthorized(req, res);
    return;
  }

  if (req.method === "GET" && url.pathname === "/sessions") {
    sendJson(req, res, 200, sessionRecords());
    return;
  }

  // Recent events (ring buffer). ?since=<seq> returns events after that
  // sequence number; ?wait=<ms> long-polls until one arrives or times out;
  // ?types=a,b filters.
  if (req.method === "GET" && url.pathname === "/events") {
    const since = Number(url.searchParams.get("since") || 0);
    const wait = Math.min(Number(url.searchParams.get("wait") || 0), 120_000);
    const types = url.searchParams.get("types")?.split(",").filter(Boolean);
    const select = () => events.filter((e) => e.seq > since && (!types || types.includes(e.type)));
    let out = select();
    if (out.length === 0 && wait > 0) {
      await new Promise((resolve) => {
        const timer = setTimeout(done, wait);
        function done() {
          clearTimeout(timer);
          eventWaiters.delete(done);
          resolve();
        }
        eventWaiters.add(done);
        req.on("close", done);
      });
      out = select();
    }
    sendJson(req, res, 200, { events: out, latestSeq: eventSeq });
    return;
  }

  // Share-state store for short LAN Tourguide links (…?state=<id>). POST a
  // viewer state, get an id; the recipient's browser GETs it back.
  if (req.method === "POST" && url.pathname === "/share-state") {
    const state = await readJson(req, res, 16 * 1024 * 1024);
    if (state === undefined) return;
    const id = crypto.randomUUID().slice(0, 12);
    try {
      saveShareState(id, state);
      sendJson(req, res, 200, { ok: true, id });
    } catch (err) {
      sendJson(req, res, 500, { ok: false, error: { message: err.message } });
    }
    return;
  }

  // Viewer-fly control channel for the embedded Python NG viewer.
  if (req.method === "POST" && url.pathname === "/viewer-fly") {
    const body = await readJson(req, res, 64 * 1024);
    if (body === undefined) return;
    lastFly = { seq: ++flySeq, ...body };
    sendJson(req, res, 200, { ok: true, seq: flySeq });
    return;
  }
  if (req.method === "GET" && url.pathname === "/viewer-fly") {
    sendJson(req, res, 200, lastFly || { seq: 0 });
    return;
  }

  if (req.method === "POST" && url.pathname === "/op") {
    const request = await readJson(req, res);
    if (request === undefined) return;
    // launch_or_attach is answered by the bridge itself: report the chosen
    // session if one is running, else signal the launcher to start one.
    if (request.op === "launch_or_attach") {
      const s = pickSession();
      if (s) sendJson(req, res, 200, { id: request.id, ok: true, result: s.record });
      else sendJson(req, res, 200, { id: request.id, ok: false, error: { message: "no running session" } });
      return;
    }
    if (STATE_OPS.has(request.op)) {
      sendJson(req, res, 200, await handleStateOp(request));
      return;
    }
    sendJson(req, res, 200, await relayOp(request));
    return;
  }

  sendJson(req, res, 404, { ok: false, error: { message: `not found: ${url.pathname}` } });
});

// --- WebSocket server (routes by path) -------------------------------------

const wss = new WebSocketServer({ noServer: true });

server.on("upgrade", (req, socket, head) => {
  const url = new URL(req.url, `http://${HOST}:${PORT}`);
  const level = originOk(req) ? authLevel(req, url) : null;
  // /agent needs the full token; /browser accepts the view token too, but a
  // view-token connection is forced read-only in handleBrowser.
  const allowed = url.pathname === "/browser" ? level !== null : level === "full";
  if (!allowed) {
    socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
    socket.destroy();
    log(`rejected WS ${url.pathname} (${!originOk(req) ? "bad origin" : "bad token"})`);
    return;
  }
  wss.handleUpgrade(req, socket, head, (ws) => {
    if (url.pathname === "/agent") handleAgent(ws);
    else if (url.pathname === "/browser") handleBrowser(ws, level === "view");
    else ws.close(1008, "unknown path");
  });
});

function handleAgent(ws) {
  agents.add(ws);
  log(`agent subscriber connected (${agents.size} total)`);
  ws.send(JSON.stringify({ seq: eventSeq, at: now(), ...connectionStatusEvent() }));
  ws.on("close", () => {
    agents.delete(ws);
    log(`agent subscriber disconnected (${agents.size} total)`);
  });
  ws.on("message", () => {
    /* agents are event subscribers; ops go over HTTP. Ignore inbound. */
  });
}

// First non-internal IPv4 (host:port) so a page can rewrite artifact / share
// URLs to the host machine instead of "localhost" — reachable by LAN/VPN
// peers when the bridge is bound to a non-loopback address. Null when bound
// to loopback (then the page keeps its connect host).
function machineHost() {
  if (LOOPBACK_HOSTS.has(HOST)) return null;
  try {
    for (const ifaces of Object.values(os.networkInterfaces())) {
      for (const i of ifaces || []) {
        if (i.family === "IPv4" && !i.internal) return `${i.address}:${PORT}`;
      }
    }
  } catch {
    /* fall through */
  }
  return null;
}

function handleBrowser(ws, viewTokenOnly) {
  let sessionId = null;
  // A ?view=1 connection registers with `viewOf` set; it is read-only and may
  // NEVER persist. Enforced here (not just client-side). A connection that
  // authenticated with the VIEW token is a viewer no matter what it claims.
  let isViewer = viewTokenOnly;
  ws.on("message", (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.kind === "register" && msg.session) {
      if (viewTokenOnly && !msg.session.viewOf) {
        ws.close(1008, "view token: read-only viewer only (open the link with ?view=1)");
        return;
      }
      sessionId = msg.session.sessionId;
      isViewer = viewTokenOnly || !!msg.session.viewOf;
      const existing = sessions.get(sessionId);
      const label = existing?.record.label ?? `workspace-${++labelCounter}`;
      const record = {
        sessionId,
        label,
        createdAt: existing?.record.createdAt ?? now(),
        lastSeenAt: now(),
        url: msg.session.url,
        mode: msg.session.mode,
        status: "running",
        readOnly: isViewer,
      };
      sessions.set(sessionId, { record, ws });
      ws.send(JSON.stringify({ kind: "registered", label, bridgeHost: machineHost() }));
      log(`browser session registered: ${label} ${sessionId} (${msg.session.mode}${isViewer ? ", read-only" : ""})`);
      publishEvent(connectionStatusEvent());
      const restoreId = msg.session.viewOf || sessionId;
      const restored = getSessionState(restoreId);
      if (restored) {
        ws.send(JSON.stringify({ kind: "restore", state: restored }));
        log(`sent restore snapshot (${restoreId}) to ${label} ${sessionId}`);
      }
    } else if (msg.kind === "persist" && msg.state) {
      if (sessionId && !isViewer) saveSessionState(sessionId, msg.state);
    } else if (msg.kind === "response" && msg.response) {
      if (isViewer) return; // a read-only viewer never answers ops
      const p = pending.get(msg.response.id);
      if (p) {
        clearTimeout(p.timer);
        pending.delete(msg.response.id);
        p.resolve(msg.response);
      }
    } else if (msg.kind === "event" && msg.event) {
      publishEvent({ ...msg.event, sessionId });
    } else if (msg.kind === "pong") {
      const s = sessionId && sessions.get(sessionId);
      if (s) s.record.lastSeenAt = now();
    }
  });
  ws.on("close", () => {
    if (sessionId && sessions.get(sessionId)?.ws === ws) {
      const s = sessions.get(sessionId);
      s.record.status = "disconnected";
      s.record.lastSeenAt = now();
      log(`browser session disconnected: ${sessionId}`);
      publishEvent(connectionStatusEvent());
    }
  });
}

// Read-only viewers must never be an op target.
const _resolveTarget = resolveTarget;
function resolveTargetWritable(sessionId) {
  const r = _resolveTarget(sessionId);
  if (r.session?.record.readOnly) return { error: `workspace tab '${r.session.record.sessionId}' is a read-only viewer` };
  return r;
}
// Route relayOp through the writable filter (liveSessions excludes viewers too).
const _liveSessions = liveSessions;
// eslint-disable-next-line no-func-assign
liveSessions = function () {
  return _liveSessions().filter((s) => !s.record.readOnly);
};
// eslint-disable-next-line no-func-assign
resolveTarget = resolveTargetWritable;

// Heartbeat: ping browsers, prune dead sessions so a new thread never attaches
// to a tab that has quietly gone away, and drop long-disconnected records.
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
  if (changed) publishEvent(connectionStatusEvent());
  publishEvent({ type: "heartbeat" });
}, PING_INTERVAL_MS);

server.listen(PORT, HOST, () => {
  log(`Tourguide Workspace bridge v${VERSION} listening on http://${HOST}:${PORT}`);
  log(`  HTTP: GET /health, GET /sessions, GET /events, POST /op, GET /artifacts/*, /share-state`);
  log(`  WS:   /browser (session), /agent (events)`);
  if (TOKEN) {
    // Never print the tokens themselves: stdout may be captured to a log file.
    log(`  auth: bearer token required — read it from ${tokenFilePath()} (mode 0600)`);
    log(`  open the workspace as: http://localhost:5173/?mode=workspace&bridgeToken=<token>`);
  } else {
    log(`  auth: DISABLED (TG_BRIDGE_NO_AUTH=1)`);
  }
  if (LOOPBACK_HOSTS.has(HOST)) {
    log(`  bound to loopback only; set TG_BRIDGE_HOST=0.0.0.0 to share sessions on the LAN`);
  }
});

process.on("exit", () => {
  if (TOKEN) {
    try {
      fs.unlinkSync(tokenFilePath());
    } catch {
      /* already gone */
    }
  }
});
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => process.exit(0));
