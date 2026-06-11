// Tiny CLI to exercise the Workspace bridge from the terminal — useful for
// validating the HTTP/WS surface without an MCP client.
//
//   node bridge/test_client.mjs health
//   node bridge/test_client.mjs sessions
//   node bridge/test_client.mjs watch                 # stream /agent events
//   node bridge/test_client.mjs op get_session
//   node bridge/test_client.mjs op fly_to '{"position":[1000,2000,3000]}'
//
// Env: TG_BRIDGE_PORT (default 7723), TG_BRIDGE_HOST (default localhost).

import crypto from "node:crypto";
import { WebSocket } from "ws";

const PORT = process.env.TG_BRIDGE_PORT || 7723;
const HOST = process.env.TG_BRIDGE_HOST || "localhost";
const BASE = `http://${HOST}:${PORT}`;
const [cmd, opName, paramsJson] = process.argv.slice(2);

async function main() {
  if (cmd === "health") {
    console.log(await (await fetch(`${BASE}/health`)).json());
  } else if (cmd === "sessions") {
    console.log(JSON.stringify(await (await fetch(`${BASE}/sessions`)).json(), null, 2));
  } else if (cmd === "watch") {
    const ws = new WebSocket(`ws://${HOST}:${PORT}/agent`);
    ws.on("open", () => console.log("[watch] subscribed to /agent events…"));
    ws.on("message", (d) => console.log("[event]", d.toString()));
    ws.on("close", () => console.log("[watch] closed"));
  } else if (cmd === "op") {
    if (!opName) throw new Error("usage: op <name> [paramsJson]");
    const body = {
      id: crypto.randomUUID(),
      op: opName,
      params: paramsJson ? JSON.parse(paramsJson) : undefined,
      source: "local_api",
    };
    const res = await fetch(`${BASE}/op`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    console.log(JSON.stringify(await res.json(), null, 2));
  } else {
    console.log("usage: health | sessions | watch | op <name> [paramsJson]");
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("error:", err.message);
  process.exit(1);
});
