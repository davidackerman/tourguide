---
title: Tourguide Analysis
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cloud analysis backend for the tourguide web-app.
---

# Tourguide Analysis Backend

The optional cloud-compute backend for the [tourguide web app](https://tourguide-8j4.pages.dev). The web app normally runs analysis browser-side via Pyodide, but Pyodide is capped at ~1.5 GB of usable input memory. When a query needs more — or wants to use the Seung-lab stack (cc3d, fastmorph, fastremap, edt, kimimaro, zmesh), or read a format Pyodide doesn't support (N5, certain precomputed scales) — the request gets routed here.

**No setup required to *use* this backend.** The web app already points at it by default. First request wakes the Space (20–60 s cold start), after which it stays hot for ~48 h.

If you want isolated compute + persistent share-link storage, [duplicate this Space](#duplicate-the-space-and-make-it-yours) into your own HF account.

---

## Supported data formats

Reads remote datasets directly — no upload — for any of:

- **zarr** v2 / v3 (via tensorstore, with zarr-python fallback for numcodecs edge cases)
- **N5** (via tensorstore's `n5` driver)
- **Neuroglancer precomputed segmentation** (via tensorstore's `neuroglancer_precomputed` driver — handles compressed_segmentation, sharded, multiscale)

URL schemes accepted: `https://`, `s3://` (anon public buckets), `gs://`. Plus `/local-data/...` via the WebSocket tunnel below — the browser bridges chunk reads from a local folder the user picked with the File System Access API.

For zarr/N5 the agent's frontend auto-picks the finest scale that fits the backend's 5 GB input budget. Precomputed-segmentation scales auto-route to the backend when their finest scale would force a downsample in the browser.

---

## API endpoints

- `GET /api/health` — heartbeat + memory info + queue depth. Used by the frontend to know when the Space is awake.
- `POST /api/analysis/run` — accepts a `CustomRequestBody` (zarr layers + optional `precomputedVolumeLayers` + Python code), loads each, runs the user's code in a sandboxed subprocess, returns table / plot / fly / new-layer outputs.
- `POST /api/inspect-source` — probes a remote zarr / N5 source for its multiscale shape (axes, scales, per-scale shape + voxel size + byte budget). Returns the same `LayerInspection` shape the browser worker emits for zarr, so the frontend's scale picker can consume it transparently. **N5 always goes through here** because zarrita (browser-side) can't read N5.
- `POST /api/share` / `GET /api/share/<id>` — short-link storage for tourguide share URLs. See [Persistent share storage](#persistent-share-storage-optional) below for the optional HF Datasets backing.
- `WS /ws/bridge/<session_id>` — tunnel for local-folder zarrs. The user's browser pushes chunk bytes over this WebSocket so the backend can analyze files that never leave the user's machine.
- `GET /api/data/<session_id>/<layer_id>/<path>` — serves synthesized zarrs that analyses create (a contact-site mask, an eroded mito volume, etc.) so the browser's Neuroglancer can load them as layers.

---

## Sandbox + security

User Python code is checked against:

- An **import whitelist** (numpy / pandas / scipy / scikit-image / matplotlib / the Seung-lab stack — and that's it; no `os`, `subprocess`, `socket`).
- A **deny-list** of dangerous patterns (file writes outside `/tmp`, shell calls, raw sockets, eval/exec on untrusted strings).

Each request runs in a forked subprocess with CPU-time and address-space `rlimit`s and a wall-clock timeout. Forking keeps numpy arrays COW-shared with the parent so layer loading is paid once across concurrent users. IP-based rate limits (10 analysis requests / minute) apply on top.

---

## Duplicate the Space and make it yours

If you want isolated compute + control over storage, copy the Space into your own HF account:

1. Open [this Space's page](https://huggingface.co/spaces/ackermand/tourguide-analysis) and click **Duplicate Space** (top-right menu). HF builds you a private copy at `https://<your-username>-tourguide-analysis.hf.space`.
2. Paste that URL into the tourguide web app: **⚙ Settings → Analysis backend → URL**.

Everything else is automatic. You now have your own rate limit, your own /tmp, and (optionally) your own persistent share storage — set up next.

---

## Persistent share storage (optional)

By default, share links created via `/api/share` are written to `/tmp` on the Space. That's **ephemeral**: `/tmp` clears on every Space restart (cold start, redeploy), so links die after a few hours-to-days. Fine for "look at this view right now"; bad for sharing in a paper or long-running collaboration.

To make share links survive Space restarts, point the Space at an HF Dataset you own. Setup is one-time:

### Step 1 — Create the dataset (~30 s)

Go to https://huggingface.co/new-dataset, fill in:
- **Owner**: your username (default)
- **Dataset name**: something like `tourguide-shares` (just a storage bucket; remember the name)
- **Visibility**: **Public** is fine — share IDs are 12-character hex (48 bits, not guessable). Private also works.

Click **Create dataset**.

### Step 2 — Generate a write token (~30 s)

Go to https://huggingface.co/settings/tokens, click **Create new token** → **Fine-grained** tab:
- **Token name**: anything (e.g., `tourguide-shares-write`)
- Scroll to **Repository permissions** → **Add scope** → select the dataset you just made → check **Write**

Click **Create token** and copy the `hf_...` string. **You won't be able to see it again** — paste it into a notes app temporarily.

### Step 3 — Add secrets to your Space (~1 min)

Go to your Space's **Settings → Variables and secrets**:

- Click **New secret**:
  - Name: `HF_TOKEN`
  - Value: paste the `hf_...` token from step 2
  - Save
- Click **New secret** (NOT variable — use secret to be safe):
  - Name: `TG_SHARE_DATASET`
  - Value: `<your-username>/<dataset-name>` (e.g., `ackermand/tourguide-shares`)
  - Save

### Step 4 — Restart the Space

**Settings → Restart this Space** (not Factory rebuild — just Restart picks up the new env vars).

After restart, check the startup log for:

```
Share storage env check: HF_TOKEN=set (hf_X…, len=37), TG_SHARE_DATASET='your-username/tourguide-shares'
Share storage: HF Datasets your-username/tourguide-shares (persistent)
```

If you see `/tmp (ephemeral …)` instead, one of the env vars didn't land — most often "the value got typed into the *description* field instead of the *value* field" when adding the secret. Edit the secret, paste the value into the right field, restart.

When persistent storage is active, share links written from the web app no longer show the expiry warning, and they keep working through Space restarts.

---

## Develop locally

```bash
cd hf-space
pip install -r requirements.txt
uvicorn app:app --port 7860 --reload
```

Point the web app at `http://localhost:7860` (Settings → Analysis backend → URL) and you've got the full backend on your own machine — same code that runs on the HF Space, but with whatever local resources you have and no rate limit.

---

## Pushing changes to your Space

This Space lives at `hf-space/` inside the main tourguide repo on GitHub. The deploy mechanism is `git subtree push` — synthesizes commits from just the `hf-space/` subdirectory and pushes them to the Space's git, where the Docker container rebuilds automatically (~3-5 min):

```bash
git subtree push --prefix hf-space \
  https://huggingface.co/spaces/<your-username>/tourguide-analysis main
```

If you've previously pushed from a different local branch, subtree's normal push complains about non-fast-forward. Use the split-then-force-push form to override:

```bash
git push https://huggingface.co/spaces/<your-username>/tourguide-analysis \
  $(git subtree split --prefix hf-space HEAD):main --force
```

Force-push is safe — the HF git is a deploy mirror, no shared collaborators.
