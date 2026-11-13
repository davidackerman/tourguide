// One-command workspace launcher: runs the Vite dev server AND the Workspace
// bridge together in a single terminal, so you don't need two windows.
//
//   npm run workspace
//
// Ctrl-C stops both. Output from both is interleaved (prefixed). For the
// truly zero-terminal flow, configure the MCP adapter with
// TOURGUIDE_WEBAPP_DIR and let `launch_or_attach` start everything.

import { spawn } from "node:child_process";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB_APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const VITE = path.join("node_modules", ".bin", "vite");
const PORT = process.env.TG_PORT || "5173";
// One bearer token for this run: handed to the bridge via env and to the
// user via the URL we print. Agents (MCP / SDK) read it from the bridge's
// token file automatically.
const TOKEN = process.env.TG_BRIDGE_TOKEN || crypto.randomBytes(24).toString("base64url");
const WORKSPACE_URL = `http://localhost:${PORT}/?mode=workspace&bridgeToken=${TOKEN}`;

const procs = [];
let shuttingDown = false;

function run(name, cmd, args, env = {}) {
  const child = spawn(cmd, args, { cwd: WEB_APP, env: { ...process.env, ...env } });
  procs.push(child);
  const prefix = (line) => `[${name}] ${line}`;
  for (const stream of [child.stdout, child.stderr]) {
    stream.setEncoding("utf8");
    let buf = "";
    stream.on("data", (chunk) => {
      buf += chunk;
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const l of lines) console.log(prefix(l));
    });
  }
  child.on("exit", (code) => {
    console.log(prefix(`exited with code ${code}`));
    shutdown(code ?? 0);
  });
  return child;
}

function shutdown(code) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const p of procs) {
    try {
      p.kill("SIGTERM");
    } catch {
      /* already gone */
    }
  }
  setTimeout(() => process.exit(code), 300);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

// --preview serves the production build (renders Neuroglancer image data
// correctly); plain dev is faster + hot-reloads but can leave image chunks
// black on some setups.
const preview = process.argv.includes("--preview");

function startServers() {
  run("bridge", "node", ["bridge/server.mjs"], { TG_BRIDGE_TOKEN: TOKEN });
  if (preview) {
    run("preview", VITE, ["preview", "--port", PORT, "--strictPort"]);
  } else {
    run("vite", VITE, ["--port", PORT, "--strictPort"]);
  }
  setTimeout(() => {
    console.log(`\n  Open the workspace:  ${WORKSPACE_URL}\n  (the token in the URL authenticates the tab to the bridge)\n`);
  }, 1500);
}

if (preview) {
  console.log("Building production bundle, then serving preview + bridge. Ctrl-C to stop.\n");
  const build = spawn("npm", ["run", "build"], { cwd: WEB_APP, stdio: "inherit" });
  build.on("exit", (code) => {
    if (code !== 0) {
      console.error(`[build] failed with code ${code}`);
      process.exit(code ?? 1);
    }
    startServers();
  });
} else {
  console.log("Starting Tourguide workspace (bridge + Vite dev). Ctrl-C to stop.\n");
  startServers();
}
