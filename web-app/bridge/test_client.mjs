// Tiny CLI to exercise the Workspace bridge from the terminal — useful for
// validating the HTTP/WS surface without an MCP client.
//
//   node bridge/test_client.mjs health
//   node bridge/test_client.mjs sessions
//   node bridge/test_client.mjs watch                 # stream /agent events
//   node bridge/test_client.mjs events [sinceSeq]     # poll recent events
//   node bridge/test_client.mjs op get_session
//   node bridge/test_client.mjs op fly_to '{"position":[1000,2000,3000]}'
//   node bridge/test_client.mjs screenshot out.png    # save the current view
//
// Env: TG_BRIDGE_PORT (default 7723), TG_BRIDGE_HOST (default localhost),
//      TG_BRIDGE_TOKEN (default: read from the bridge's token file).

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { WebSocket } from "ws";

const PORT = process.env.TG_BRIDGE_PORT || 7723;
const HOST = process.env.TG_BRIDGE_HOST || "localhost";
const BASE = `http://${HOST}:${PORT}`;
const [cmd, arg1, arg2] = process.argv.slice(2);

function readToken() {
  if (process.env.TG_BRIDGE_TOKEN) return process.env.TG_BRIDGE_TOKEN;
  try {
    const raw = fs.readFileSync(path.join(os.tmpdir(), `tourguide-bridge-${PORT}.token`), "utf8").trim();
    return raw.startsWith("{") ? JSON.parse(raw).token || "" : raw; // {token, viewToken}
  } catch {
    return "";
  }
}
const TOKEN = readToken();
const authHeaders = TOKEN ? { authorization: `Bearer ${TOKEN}` } : {};

async function getJson(pathname) {
  const res = await fetch(`${BASE}${pathname}`, { headers: authHeaders });
  return res.json();
}

async function postOp(op, params) {
  const body = { id: crypto.randomUUID(), op, params, source: "local_api" };
  const res = await fetch(`${BASE}/op`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function main() {
  if (cmd === "health") {
    console.log(await (await fetch(`${BASE}/health`)).json());
  } else if (cmd === "sessions") {
    console.log(JSON.stringify(await getJson("/sessions"), null, 2));
  } else if (cmd === "events") {
    console.log(JSON.stringify(await getJson(`/events?since=${arg1 || 0}`), null, 2));
  } else if (cmd === "watch") {
    const ws = new WebSocket(`ws://${HOST}:${PORT}/agent${TOKEN ? `?token=${TOKEN}` : ""}`);
    ws.on("open", () => console.log("[watch] subscribed to /agent events…"));
    ws.on("message", (d) => console.log("[event]", d.toString()));
    ws.on("close", (code) => console.log("[watch] closed", code));
    ws.on("error", (e) => console.error("[watch] error", e.message));
  } else if (cmd === "op") {
    if (!arg1) throw new Error("usage: op <name> [paramsJson]");
    console.log(JSON.stringify(await postOp(arg1, arg2 ? JSON.parse(arg2) : undefined), null, 2));
  } else if (cmd === "screenshot") {
    const out = arg1 || "screenshot.png";
    const res = await postOp("screenshot", { maxWidth: 1600 });
    if (!res.ok) throw new Error(res.error?.message || "screenshot failed");
    fs.writeFileSync(out, Buffer.from(res.result.png, "base64"));
    console.log(`wrote ${out} (${res.result.width}x${res.result.height})`);
  } else {
    console.log("usage: health | sessions | events [since] | watch | op <name> [json] | screenshot [file]");
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
