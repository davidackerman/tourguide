# Tourguide (web app)

A 3D microscopy viewer with built-in structured-data browsing, natural-language queries, and Python analysis — Neuroglancer embedded in the page, an AI agent that turns "measure the mitochondria" into running code, and share links that survive a browser restart. No install required.

**Live**: https://tourguide-8j4.pages.dev

This README is the user-feature guide. For the optional cloud-compute backend, see [`../hf-space/README.md`](../hf-space/README.md). For development / building / deploying your own copy, scroll to the bottom.

---

## What it does

- **Loads remote datasets directly** — zarr v2/v3, N5, Neuroglancer precomputed (segmentations, meshes, skeletons). Sources can be on S3, GCS, public HTTPS, or a local folder picked via the browser's File System Access API.
- **Embedded Neuroglancer** — full NG renderer in an iframe, with live state tracking (position, zoom, selected segments).
- **Structured browser** — point-and-click filtering of organelle CSVs (volume / surface area / COM / etc.) with click-to-fly into the viewer.
- **Natural-language queries** — ask in plain English. Examples:
  - *"show me the largest 10 mitochondria"*
  - *"plot the volume distribution"*
  - *"measure properties of mito_seg"* — runs `cc3d.statistics` on the segmentation volume, ingests results into the SQL database, click any row to fly the viewer there
  - *"how many neurons are there?"* — reads from `segment_properties` metadata, no compute
- **Agent-generated Python** — `python_on_layers` loads layer voxels as numpy arrays (in-browser Pyodide for small, HF Space for big), runs your code, surfaces the results as a table / plot / new layer.
- **Share links** — Copy a URL that round-trips the dataset + viewer state + computed tables. If a backend is configured, links shorten to `?s=<id>` and (with HF Datasets storage) persist across Space restarts.
- **Copy NG link** — Plain Neuroglancer permalink with just the viewer state, no tourguide data. Useful for Slack / papers / handing a view to someone who doesn't use tourguide.
- **Runs fully on-prem if you want** — see [Running everything locally](#running-everything-locally).

---

## Quick start

1. Go to the [live URL](https://tourguide-8j4.pages.dev).
2. Click the welcome dialog's **Load demo dataset** to pull a public HeLa-2 sample, or paste a YAML descriptor / NG state URL.
3. Click **⚙ Settings** and configure an AI backend (see next section) — required only if you want to use natural-language queries; the structured browser works without one.
4. Start asking questions in the **Ask** box on the right panel.

---

## Setting up an AI backend

Pick **one** of these. Free options listed first.

| Provider | Cost | Speed | Notes |
|---|---|---|---|
| **Gemini** (free tier) | $0 | Fast | ~500 req/day on `gemini-3.1-flash-lite-preview`. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Easiest start. |
| **WebLLM** (in-browser) | $0 | Slow | Runs the model locally on your GPU via WebGPU. ~1 GB one-time download per model. Fully offline. |
| **Anthropic Claude** | Paid | Fast | $3/M input / $15/M output for Sonnet 4.6 (default); Haiku 4.5 is ~3× cheaper. Built-in prompt caching cuts agent-loop cost ~10× on cached tokens. Get a key at [console.anthropic.com](https://console.anthropic.com). |
| **OpenAI** | Paid | Fast | Pick any model. |
| **OpenRouter** | Paid | Fast | Single key, ~100 models from Claude / Gemini / Llama / etc. |
| **xAI Grok** | Paid | Fast | One key, OpenAI-compatible. |
| **OpenAI-compatible (custom)** | Varies | Varies | Point at any OpenAI-style endpoint — local Ollama, vLLM, LM Studio, llama.cpp server, on-prem GPU cluster. |

Settings → **AI backend** → pick a provider → paste your key → **Test key**. The Claude / Gemini sections include a **Refresh available models** button that pulls the real model list from your account's `/v1/models` endpoint, so you're never picking from a hardcoded stale list.

Keys are stored in `localStorage` on your machine only — tourguide's frontend is static, no server we control sees them.

---

## Analysis backend (optional)

For datasets too large to fit in Pyodide's ~1.5 GB browser memory (think hemibrain at 512 nm), the agent can route analysis to a Hugging Face Space that has tensorstore, the full Seung-lab stack (cc3d, fastmorph, fastremap, edt, kimimaro, zmesh), and ~16 GB RAM.

You don't have to configure anything to use the default shared backend (`ackermand-tourguide-analysis.hf.space`) — it's preconfigured. It's rate-limited and may cold-start (~30-60 s if it's been idle), but works for casual use.

For real work, **duplicate the Space** into your own free HF account so you have isolated compute + storage. See [`../hf-space/README.md`](../hf-space/README.md) for setup + the optional persistent share-link storage.

---

## Running everything locally

You can run the entire stack on your own machine — no cloud — if you want fully on-prem operation:

| Cloud version | Local equivalent |
|---|---|
| Cloudflare Pages (frontend) | `npm run build && npm run preview` — serves on `http://localhost:4173` |
| HF Space (analysis backend) | `cd ../hf-space && uvicorn app:app --port 7860` — same code as the Space |
| Gemini / Claude / OpenAI API (LLM) | Local Ollama via the OpenAI-compatible backend (`http://localhost:11434/v1`), or in-browser WebLLM |
| S3 / GCS / HTTPS (data) | Local folder via the browser's File System Access API (Settings → Load my data → folder picker) |
| HF Datasets (persistent shares) | `/tmp` on your own machine — automatic fallback when `HF_TOKEN` isn't set |

For an air-gapped workstation: `npm run preview` + `uvicorn` + Ollama + local files. All four pieces talk to each other over `localhost`; nothing leaves the machine.

---

## Adding your own dataset

Two ways:

1. **Paste an NG state URL** in the loader — works for any Neuroglancer permalink. Tourguide infers the descriptor from the state.
2. **Write a YAML descriptor** under `public/datasets/` and reference it from `public/catalog.json`. See `public/datasets/jrc_hela-2.yaml` for the schema. Required fields: `name`, `display_name`, `voxel_size_nm`, `layers`. Each layer needs `name`, `type` (`image` or `segmentation`), `source` (NG source URL — `zarr://`, `n5://`, `precomputed://`, etc.).

The descriptor approach lets you bundle organelle CSVs (volume / COM / surface area columns) so the structured browser + NL queries work without extra setup. See [`../CSV_COLUMN_GUIDE.md`](../CSV_COLUMN_GUIDE.md) for the column conventions.

---

## Develop & build

```bash
cd web-app
npm install
npm run dev       # vite dev server, ~5173
npm run build     # production bundle into dist/
npm run preview   # serve the production bundle locally
```

Type-checking runs as part of `build`. To check without building: `npx tsc --noEmit`.

---

## Deploy your own copy

**Recommended: Cloudflare Pages.** No bandwidth cap, no commercial-use restrictions, custom headers work. Setup:

1. [dash.cloudflare.com](https://dash.cloudflare.com) → Workers & Pages → Create → Pages → Connect to Git.
2. Pick the repo. **Root directory:** `web-app`. **Framework preset:** Vite (or set manually: build command `npm run build`, output `dist`).
3. Deploy. URL is `https://<project>.pages.dev`. Every PR gets a preview URL. Custom domain support is free.

`public/_headers` applies COOP/COEP cross-origin isolation, which WebLLM needs for full-speed in-browser inference. `public/_redirects` makes deep-link share URLs work as a single-page app.

**Alternative: Vercel.** Same idea, slightly more polished UX, non-commercial use only on free tier. Import at [vercel.com/new](https://vercel.com/new), Root Directory `web-app`, Vite preset auto-detected. Headers come from `vercel.json`.

**Alternative: GitHub Pages.** A workflow at `.github/workflows/pages.yml` deploys on push. In repo Settings → Pages → Source, pick **GitHub Actions**. Note: GH Pages can't set COOP/COEP headers, so WebLLM may run slower or fail there. Use Cloudflare or Vercel for the local-AI tier.

**Local-network testing**: WebGPU and `navigator.userAgentData` require a secure context. `http://localhost` counts; `http://192.168.x.x:5173` does NOT. Either stick to localhost, or run `cloudflared tunnel --url http://localhost:5173` for an `https://…` URL.
