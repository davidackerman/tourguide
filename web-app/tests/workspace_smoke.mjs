// End-to-end smoke test for Tourguide workspace mode.
//
// Self-contained: starts the bridge + Vite, drives a real headless Chromium
// to /?mode=workspace, and asserts the full loop:
//   1. the tab connects to the bridge (connection dot turns green),
//   2. the bridge lists exactly one running workspace session,
//   3. get_session round-trips through the live page,
//   4. a write op (save_session_state) round-trips AND shows up in the
//      Agent Actions panel.
//
// Run:  npm run test:smoke
// No dataset is loaded (no network/WebGL data dependency); the ops exercised
// don't require a mounted viewer, so this stays deterministic in CI.

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { chromium } from "playwright";

const WEB_APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const BRIDGE_PORT = process.env.SMOKE_BRIDGE_PORT || "7790";
const WEB_PORT = process.env.SMOKE_WEB_PORT || "5179";
const BRIDGE = `http://localhost:${BRIDGE_PORT}`;
const PAGE_URL = `http://localhost:${WEB_PORT}/?mode=workspace&bridgePort=${BRIDGE_PORT}`;

const children = [];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function spawnChild(cmd, args, opts) {
  const c = spawn(cmd, args, { cwd: WEB_APP, stdio: "ignore", ...opts });
  children.push(c);
  return c;
}

async function waitFor(label, fn, timeoutMs = 90_000, interval = 500) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      if (await fn()) return;
    } catch {
      /* keep polling */
    }
    await sleep(interval);
  }
  throw new Error(`timed out waiting for: ${label}`);
}

async function postOp(op, params) {
  const res = await fetch(`${BRIDGE}/op`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id: `smoke-${op}`, op, params, source: "local_api" }),
  });
  return res.json();
}

function assert(cond, msg) {
  if (!cond) throw new Error(`ASSERT FAILED: ${msg}`);
  console.log(`  ✓ ${msg}`);
}

async function main() {
  console.log("[smoke] starting bridge…");
  spawnChild("node", ["bridge/server.mjs"], { env: { ...process.env, TG_BRIDGE_PORT: BRIDGE_PORT } });
  await waitFor("bridge /health", async () => (await (await fetch(`${BRIDGE}/health`)).json()).ok);

  console.log("[smoke] starting Vite…");
  spawnChild(path.join("node_modules", ".bin", "vite"), ["--port", WEB_PORT, "--strictPort"]);
  await waitFor("vite server", async () => (await fetch(`http://localhost:${WEB_PORT}/`)).ok);

  console.log("[smoke] launching headless Chromium…");
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--use-gl=swiftshader", "--enable-unsafe-swgl"],
  });
  try {
    const page = await browser.newPage();
    page.on("pageerror", (e) => console.warn("  [page error]", e.message));
    await page.goto(PAGE_URL, { waitUntil: "domcontentloaded" });

    console.log("[smoke] asserting live connection + ops…");
    await page.waitForSelector(".conn-dot.conn-connected", { timeout: 30_000 });
    assert(true, "workspace tab connected to bridge (dot is green)");

    const sessions = await (await fetch(`${BRIDGE}/sessions`)).json();
    const running = sessions.filter((s) => s.status === "running" && s.mode === "workspace");
    assert(running.length === 1, `bridge lists one running workspace session (got ${running.length})`);

    const gs = await postOp("get_session");
    assert(gs.ok === true, "get_session returned ok");
    assert(gs.result?.mode === "workspace", "get_session reports mode=workspace");

    const saved = await postOp("save_session_state", { name: "smoke" });
    assert(saved.ok === true && typeof saved.result?.id === "string", "save_session_state returned an id");

    // The Agent Actions panel should show the write op (read-only ops like
    // get_session are intentionally NOT logged).
    await page.waitForFunction(
      () => [...document.querySelectorAll(".action-name")].some((e) => e.textContent === "save_session_state"),
      { timeout: 10_000 },
    );
    assert(true, "Agent Actions panel shows the save_session_state entry");

    const names = await page.$$eval(".action-name", (els) => els.map((e) => e.textContent));
    assert(
      !names.includes("get_session"),
      "read-only get_session is NOT in the action history",
    );

    console.log("\n[smoke] PASS ✅");
  } finally {
    await browser.close();
  }
}

main()
  .then(() => teardown(0))
  .catch((err) => {
    console.error("\n[smoke] FAIL ❌", err.message);
    teardown(1);
  });

function teardown(code) {
  for (const c of children) {
    try {
      c.kill("SIGTERM");
    } catch {
      /* ignore */
    }
  }
  // Give children a moment to exit, then hard-exit.
  setTimeout(() => process.exit(code), 500);
}
