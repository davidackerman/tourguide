// One-command workspace launcher: runs the Vite dev server AND the Workspace
// bridge together in a single terminal, so you don't need two windows.
//
//   npm run workspace
//
// Ctrl-C stops both. Output from both is interleaved (prefixed). For the
// truly zero-terminal flow, configure the MCP adapter with
// TOURGUIDE_WEBAPP_DIR and let `launch_or_attach` start everything.

import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB_APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const VITE = path.join("node_modules", ".bin", "vite");

const procs = [];
let shuttingDown = false;

function run(name, cmd, args) {
  const child = spawn(cmd, args, { cwd: WEB_APP });
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

console.log("Starting Tourguide workspace (bridge + Vite). Ctrl-C to stop.\n");
run("bridge", "node", ["bridge/server.mjs"]);
run("vite", VITE, []);
