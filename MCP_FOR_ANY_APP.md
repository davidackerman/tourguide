# Driving any program from Claude with MCP

A practical guide to giving an LLM agent (Claude Desktop, Claude Code, Cursor,
…) control of an existing program — written from two real implementations:

- **`neuroglancer-mcp`** — the MCP server *embeds* a Neuroglancer viewer.
- **Tourguide** — the MCP server *bridges* to a separately-running web app.

If you have a program (say, something built on Neuroglancer **and** Blender)
and you want to "run it from Claude and let Claude talk to it," this is the
map.

---

## 1. The mental model

```
Agent  (Claude / Cursor / a script)        owns: conversation, reasoning, compute
   │  MCP (stdio JSON-RPC)
   ▼
Adapter (your MCP server)                  owns: translating intent → app actions
   │
   ▼
Program (Neuroglancer, Blender, your app)  owns: its domain + visual/file state
```

Three rules that keep this clean:

1. **The agent owns reasoning and (if it can) compute.** Don't rebuild an LLM
   or a Python runtime inside your program — the agent already is one.
2. **The program owns its state** — the 3D scene, the camera, the meshes, the
   files. The agent doesn't hold that; it asks the program to change it.
3. **The MCP server is a thin adapter**, not the product. Anything genuinely
   useful belongs in your program's own API; MCP is just the first doorway to
   it. (A second doorway — a Python SDK, an HTTP client — should be able to
   call the same operations.)

The single most important question is everything below: **does the MCP server
*embed* the program, or *bridge* to it?**

---

## 2. The two architectures

### Pattern A — Embed (the MCP server *is* the program's driver)

The MCP server process imports the program as a library and holds its state in
the same process.

```
Claude ──stdio──► MCP server  ──in-process──►  program's API  ──►  a viewer URL / window
                  (holds the live object)
```

**`neuroglancer-mcp` does this.** It depends on the `neuroglancer` Python
package, constructs a viewer object in-process, and each tool call mutates that
object. The first call returns a URL; you open it once and every later call
updates that live viewer. One process = MCP + program.

**Use when:** the program has a real **library / API you can drive in-process**
(Python `neuroglancer`, a headless engine, an SDK). Simplest possible setup —
no extra server, no ports.

**Limits:** the program must be embeddable. A full GUI app you click around in
(Blender, a desktop app, a browser tab a human is using) usually *can't* be
embedded in your MCP process — its state lives in *its* process, not yours.

### Pattern B — Bridge (the MCP server talks to a separately-running program)

The program runs on its own (a GUI, a browser tab, Blender). The MCP server is
a thin proxy that relays commands to it over a local socket.

```
Claude ──stdio──► MCP server ──HTTP/WS──► bridge ──WS/socket──► running program
                  (thin proxy)            (hub)                  (source of truth)
```

**Tourguide does this.** A browser tab can't host a server, but the live
viewer/data live in the browser. So a tiny local **bridge** process is the hub:
the app connects to it and registers a session; the MCP server POSTs operations
to the bridge, which relays them to the app and streams events back. The app is
the source of truth; the MCP server holds only lightweight session metadata.

**The Blender MCP pattern is the same shape:** a Blender **add-on** runs *inside*
Blender and listens on a socket; the MCP server sends it commands; Blender (the
real app you're watching) executes them. You can't import Blender's `bpy` from
outside, so you inject a listener *into* the running app and talk to it.

**Use when:** the program is a long-running **GUI you watch**, a **browser app**,
or anything whose state can't live in your MCP process. This is the common case
for "I want to see it happen live."

### Picking one

| Your program is… | Pattern |
| --- | --- |
| A library/SDK you can drive headless in-process | **A — Embed** |
| A GUI app (Blender, a desktop tool) you watch | **B — Bridge** (add-on/socket inside it) |
| A web app in a browser tab | **B — Bridge** (app ↔ local bridge ↔ MCP) |
| A mix (e.g. Neuroglancer **and** Blender) | usually **B for the GUI parts**, A for the embeddable parts — see §6 |

---

## 3. The transport, concretely

- **Agent ↔ MCP server:** always **stdio JSON-RPC** (the MCP standard). Your
  client config launches the server as a subprocess; you write tools, the SDK
  handles the protocol. Use the official `mcp` Python SDK (`FastMCP`) — a tool
  is just a decorated function.
- **MCP server ↔ running program (Pattern B only):** your choice, but
  **HTTP for request/response + WebSocket for live events** is a clean default.
  HTTP carries `do_thing → result`; WebSocket carries the program's live state /
  event stream and lets the bridge push commands into a browser/GUI that can't
  be polled.
- **Why a separate bridge process for browser/GUI apps:** a browser tab or a
  Blender add-on can't *host* a server the MCP can dial into reliably, and
  multiple things may want to attach. A small always-on bridge owns the socket,
  relays messages, and survives the MCP server restarting.

Don't invent an MCP-specific internal protocol. Define your program's
operations once (an "app API"); let MCP be a thin translation of stdio calls
into those operations.

---

## 4. A build checklist

1. **List ~15–25 *semantic* operations**, not every low-level primitive.
   `select_objects`, `fly_to`, `add_mesh`, `run_query`, `snapshot` — not
   `set_pixel`, `set_matrix_element`. Add one **escape hatch** (`set_raw_state`)
   for the long tail. (40 micro-tools overwhelm the agent and you.)
2. **Define those operations as a plain API** in the program (a function table /
   HTTP routes). This is the durable artifact.
3. **Pattern A:** wrap the library in the MCP server, one tool per operation.
   **Pattern B:** stand up the bridge (HTTP `/op` + WS events), make the program
   connect + register, and make the MCP server a thin client that forwards each
   tool call to `/op`.
4. **Launch / attach** (load-bearing): on the first tool call, if the program is
   already running, attach to it; if not, start it (and the bridge); if several
   sessions exist, pick one deterministically; on disconnect, reconnect or error
   clearly. Don't make the agent babysit this.
5. **Write good tool docstrings.** They're the agent's API docs — say what each
   tool does, its args, and when to use it vs. another.
6. **Decide where compute lives** (see §5).
7. **Test headless.** Drive the whole stack from a script (and, for a browser
   app, a real headless browser) so you can prove `op → result → visible change`
   without clicking.

---

## 5. Where does the *compute* run? (the part people get wrong)

If a request needs real work — "measure every object," "generate meshes" —
decide who does it:

- **The agent has a shell (Claude Code, Cursor, a script):** let *it* compute.
  It reads the data, runs real Python (numpy/scipy/your libs), and calls a thin
  **`ingest_result`** tool to push the answer into the program. The MCP server
  stays thin; no compute engine inside your app. **This is the ideal when your
  user is on a coding agent.**
- **The agent has no shell (Claude Desktop, ChatGPT Desktop):** it can only call
  your tools — so the **compute must live in your MCP server** (real local
  Python) or a backend you control. Ship a `run_python(code, inputs)` tool or
  canned operations. More setup, but works for non-coders.

Key realization: **Claude Desktop cannot run local Python** — it only has your
MCP tools (plus a JS sandbox that can't import your libraries or read your
files). Claude Code / Cursor *can*. So your target client decides whether the
MCP server needs "hands" or can stay a pure bridge.

---

## 6. The friend's case: a window showing Neuroglancer + Blender snapshots

It matters *how* Blender is used, and there are two very different setups. Pick
by how Blender runs:

### 6a. Blender is driven **via Python** and shows **rendered snapshots** (the friend's actual case)

He has **one window** that embeds Neuroglancer and displays **rendered images
from Blender**, where Blender is run **headless via Python** (`bpy` as a module,
or `blender --background --python render.py`). Then Blender is **not a GUI to
socket into — it's compute that emits PNGs.** This is the *simpler* case and
maps almost exactly onto Tourguide:

- **The window** (NG + snapshot panel) is the visual app → **Pattern B bridge**,
  exactly like Tourguide. (NG inside it is part of that window's state.)
- **Blender** is a **render step**: run the Python, get a PNG, and push it into
  the window with an artifact op — `show_snapshot(png)` / `ingest_image(png)`
  — the direct analog of Tourguide's `show_plot(png)`.
- **Who runs the Blender Python?**
  - *Claude Code / a script:* the agent runs his Blender script itself, gets the
    PNG, calls `show_snapshot`. The MCP server stays a pure bridge. (Ideal.)
  - *Claude Desktop (no shell):* the **MCP server** runs Blender (it's just
    `subprocess`/`bpy`) via a `render_blender(params)` op and displays the
    result. Since he "just runs Blender via Python" already, this is a few lines.

```
                                        push computed artifacts in
Claude ──stdio──► MCP ──HTTP/WS──► bridge ──WS──► the window (Neuroglancer + snapshot panel)
   │  runs render.py (bpy) ──► snapshot.png ──ingest_image──┘
```

So: **one bridge (to the window)**, and Blender is just compute feeding
`show_snapshot`. No Blender add-on, no socket, no second bridge.

### 6b. Blender is an **interactive GUI you click around in**

*Different* situation — if instead you want to drive a Blender app a human has
open on screen, then Blender *is* a live GUI: ship a small **add-on inside
Blender** that listens on a socket and runs ops via `bpy` on the main thread
(the popular "Blender MCP" pattern, = Pattern B for Blender too). Use this only
if you genuinely need the live interactive session — for "render and show me,"
6a is simpler and is what he's doing.

In both, **the agent reasons/computes; the window owns the visual state; the MCP
server routes intent + relays artifacts.**

---

## 7. Design principles (learned the hard way)

- **Thin adapter, fat API.** If you're tempted to add a convenience to the MCP
  server, add it to the program's API instead; MCP just exposes it.
- **Semantic tools + one escape hatch.** Don't mirror the program's entire
  low-level surface.
- **Layer it:** `Program API ← MCP adapter ← (Python SDK, HTTP client, future
  protocol)`. MCP is young and may be replaced; if your operations live in an
  app API, swapping the adapter is cheap.
- **Make launch/attach deterministic**, and **warn on ambiguity** (e.g. two app
  windows open → which one does a command hit?).
- **Record what the agent did** (an action log in the app) — invaluable for
  trust and debugging, and it's not the same as the chat transcript.
- **One window/session.** Multiple live instances + "use the most recent" is a
  silent footgun; surface it.

---

## 8. Would a reusable tool/repo for this be worth building?

**Honest take: yes, but scope it carefully — the boilerplate isn't the hard part.**

What's *already* easy (don't rebuild it): the MCP protocol itself. `FastMCP`
(Python) and the TS SDK already make "function → tool" trivial. A framework that
just wraps that adds little.

What's *actually* hard, repeatedly, and worth a tool:

- **The Pattern-B bridge.** A reusable local **HTTP+WS hub** with
  session registry, launch/attach, reconnect, request relay, and an event
  stream — plus thin client libs for both sides (the app and the MCP server).
  That's ~the thing we hand-rolled in Tourguide and would gladly not write
  again. **This is the strongest candidate for a library.**
- **App-side connectors:** a tiny "register + handle ops + emit events" client
  for a **browser** (JS/TS) and for a **GUI add-on** (a Blender/Python
  template). Most of the per-project pain is here.
- **Launch/attach + a session model** as a drop-in (the load-bearing,
  fiddly-to-get-right part).
- **A scaffold/CLI:** `create-app-mcp` that generates the MCP server, the
  bridge, an app connector stub, and a headless smoke test — so a new program is
  "fill in your operations," not "design the whole transport."
- **Conventions for the result-sink pattern** (`ingest_table`, `add_layer`,
  `show_image`) so agent-computed artifacts flow back uniformly.

Risks / why to be careful:

- **MCP is moving fast** — pin to the protocol, keep the bridge protocol-agnostic
  so you survive churn.
- **Don't over-abstract the operations** — every program's semantics differ; the
  tool should make the *plumbing* trivial and stay out of the *domain*.
- **Security:** `run_python`/op relays execute things on the user's machine —
  bake in optional sandboxing and clear trust boundaries.

So: a **"bridge + connectors + scaffold" toolkit** (not "yet another MCP
framework") would genuinely save real work and standardize the launch/attach +
live-event parts that everyone re-implements. The Tourguide bridge
(`web-app/bridge/`) and the layered Workspace API are basically a prototype of
exactly that — extracting them into a generic, app-agnostic package is a
reasonable next project.

---

### TL;DR

- Decide **embed vs bridge** first. GUI/browser → bridge; library/SDK → embed.
- MCP server = **thin**; define your program's operations as a real API.
- Use **stdio** to the agent, **HTTP+WS** to a separately-running app.
- Nail **launch/attach** and **one-session** clarity.
- Let the **agent compute** when it can (Claude Code); give the **server hands**
  only when it can't (Claude Desktop).
- A reusable **bridge + connectors + scaffold** would help; a generic MCP
  framework wouldn't (that already exists).
