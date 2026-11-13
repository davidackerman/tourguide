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
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright";

const WEB_APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const BRIDGE_PORT = process.env.SMOKE_BRIDGE_PORT || "7790";
const WEB_PORT = process.env.SMOKE_WEB_PORT || "5179";
const BRIDGE = `http://localhost:${BRIDGE_PORT}`;
// The bridge requires a bearer token; we pick one here, hand it to the
// bridge via env, to the tab via the URL, and to our own requests via header.
const TOKEN = "smoke-" + Math.random().toString(36).slice(2);
const PAGE_URL = `http://localhost:${WEB_PORT}/?mode=workspace&bridgePort=${BRIDGE_PORT}&bridgeToken=${TOKEN}`;
const AUTH = { authorization: `Bearer ${TOKEN}` };

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

async function postOp(op, params, headers = AUTH) {
  const res = await fetch(`${BRIDGE}/op`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify({ id: `smoke-${op}-${Date.now()}`, op, params, source: "local_api" }),
  });
  return { status: res.status, ...(await res.json()) };
}

function assert(cond, msg) {
  if (!cond) throw new Error(`ASSERT FAILED: ${msg}`);
  console.log(`  ✓ ${msg}`);
}

async function main() {
  console.log("[smoke] starting bridge…");
  // Keep the bridge's disk state (saved states, session snapshots, share
  // blobs, artifacts) in a scratch dir so the test never touches ~/.tourguide.
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "tg-smoke-"));
  spawnChild("node", ["bridge/server.mjs"], {
    env: {
      ...process.env,
      TG_BRIDGE_PORT: BRIDGE_PORT,
      TG_BRIDGE_TOKEN: TOKEN,
      TG_STATE_DIR: path.join(scratch, "saved-states"),
      TG_SESSION_STATE_DIR: path.join(scratch, "session-states"),
      TG_SHARE_STATE_DIR: path.join(scratch, "shared-states"),
      TG_ARTIFACTS_DIR: path.join(scratch, "artifacts"),
    },
  });
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

    // Auth: no token -> 401; wrong Origin -> 403; correct token -> 200.
    const noAuth = await fetch(`${BRIDGE}/sessions`);
    assert(noAuth.status === 401, "GET /sessions without a token is rejected (401)");
    const badOrigin = await fetch(`${BRIDGE}/sessions`, { headers: { ...AUTH, origin: "https://evil.example" } });
    assert(badOrigin.status === 403, "requests from a non-loopback Origin are rejected (403)");
    const unauthOp = await postOp("get_session", undefined, {});
    assert(unauthOp.status === 401, "POST /op without a token is rejected (401)");

    const sessions = await (await fetch(`${BRIDGE}/sessions`, { headers: AUTH })).json();
    const running = sessions.filter((s) => s.status === "running" && s.mode === "workspace");
    assert(running.length === 1, `bridge lists one running workspace session (got ${running.length})`);

    const gs = await postOp("get_session");
    assert(gs.ok === true, "get_session returned ok");
    assert(gs.result?.mode === "workspace", "get_session reports mode=workspace");

    const saved = await postOp("save_session_state", { name: "smoke" });
    assert(saved.ok === true && typeof saved.result?.id === "string", "save_session_state returned an id");
    assert(typeof saved.result?.path === "string" && saved.result.path.startsWith(scratch), "saved state was written to the bridge's state dir");

    // A read-only viewer connection: the VIEW token must not unlock /sessions.
    const viewTok = await fetch(`${BRIDGE}/sessions`, { headers: { authorization: "Bearer not-the-token" } });
    assert(viewTok.status === 401, "a wrong token is rejected on /sessions (401)");

    // ingest_table: push agent-computed rows in, then prove they landed in the
    // real in-browser DB by querying them back.
    const ing = await postOp("ingest_table", {
      name: "smoke_mito",
      columns: ["object_id", "volume_nm_3"],
      rows: [[1, 100], [2, 200], [3, 300]],
    });
    assert(ing.ok === true && ing.result?.tableId === "smoke_mito", "ingest_table returned the table id");
    const q = await postOp("run_sql", { sql: "SELECT COUNT(*) AS n, SUM(volume_nm_3) AS s FROM smoke_mito" });
    assert(q.ok === true && q.result?.rows?.[0]?.[0] === 3, "ingested rows are queryable (count=3)");
    assert(q.result?.rows?.[0]?.[1] === 600, "ingested values are correct (sum=600)");
    const gs2 = await postOp("get_session");
    assert(
      (gs2.result?.tables ?? []).some((t) => t.id === "smoke_mito"),
      "ingested table shows up in get_session",
    );

    // run_sql is read-only: a write must be refused and leave the table intact.
    const bad = await postOp("run_sql", { sql: "DROP TABLE smoke_mito" });
    assert(bad.ok === false && /read-only/.test(bad.error?.message ?? ""), "run_sql refuses a DROP");
    const still = await postOp("run_sql", { sql: "SELECT COUNT(*) FROM smoke_mito" });
    assert(still.ok === true && still.result?.rows?.[0]?.[0] === 3, "table survived the refused write");

    // show_plot takes a PNG the agent rendered (1x1 transparent PNG here).
    const PNG_1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
    const plot = await postOp("show_plot", { png: PNG_1x1, title: "smoke plot" });
    assert(plot.ok === true && typeof plot.result?.id === "string", "show_plot accepted a PNG and returned an id");
    const rejected = await postOp("show_plot", { code: "plt.plot([1,2])" });
    assert(rejected.ok === false, "show_plot without a png is refused (no in-browser compute)");

    // wait_for_ready with no viewer mounted returns promptly with ready=false.
    const wr = await postOp("wait_for_ready", { timeoutMs: 300 });
    assert(wr.ok === true && wr.result?.ready === false, "wait_for_ready times out cleanly with no dataset");

    // Events: the action stream is exposed over GET /events for pollers.
    const ev = await (await fetch(`${BRIDGE}/events?since=0`, { headers: AUTH })).json();
    assert(
      ev.events.some((e) => e.type === "action" && e.entry?.action === "ingest_table"),
      "GET /events contains the ingest_table action",
    );
    assert(ev.events.some((e) => e.type === "connection_status"), "GET /events contains connection_status");

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
