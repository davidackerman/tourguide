# tourguide-web

Static web version of tourguide. Open it in a browser, pick a dataset,
query it in plain English, navigate to results, and share the view as a
URL. No install required for users with publicly-hosted data.

## Status

Milestone 1 — MVP shell. Loads HeLa-2 from the public
janelia-cosem-datasets S3 bucket via an embedded Neuroglancer iframe,
driven by a YAML dataset descriptor and a curated `catalog.json`.

## Develop

```bash
cd web-app
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

## Build

```bash
npm run build
npm run preview
```

The production bundle goes to `dist/`.

## Deploy

**Recommended: Cloudflare Pages.** No bandwidth cap, no
commercial-use restrictions, custom headers work. Setup:

1. [dash.cloudflare.com](https://dash.cloudflare.com) → Workers & Pages → Create → Pages → Connect to Git.
2. Pick the repo. **Root directory:** `web-app`. **Framework preset:** Vite (or set manually: build command `npm run build`, output `dist`).
3. Deploy. URL is `https://<project>.pages.dev`. Every PR gets a preview URL.

The `public/_headers` file in this repo automatically applies
COOP/COEP cross-origin isolation, which WebLLM needs for full-speed
in-browser inference. The `public/_redirects` file makes deep links
work as a single-page app.

**Alternative: Vercel.** Same idea, slightly more polished UX,
non-commercial use only on the free tier. Import at
[vercel.com/new](https://vercel.com/new), Root Directory `web-app`,
Vite preset auto-detected. Headers come from `vercel.json`.

**Alternative: GitHub Pages.** A workflow at
`.github/workflows/pages.yml` deploys on push. In repo Settings → Pages
→ Source, pick **GitHub Actions**. Note: GH Pages can't set COOP/COEP
headers, so WebLLM may run slower or fail there. Use Cloudflare or
Vercel if you want the no-key local-AI tier to work at full speed.

**Local network access** (e.g. testing from another laptop): WebGPU
and `navigator.userAgentData` require a secure context.
`http://localhost` counts; `http://192.168.x.x:5173` does NOT. Either
stick to localhost, or run a tunnel like `cloudflared tunnel --url
http://localhost:5173` to get an `https://…` URL.

## Adding a dataset

Add a YAML descriptor under `public/datasets/` and reference it from
`public/catalog.json`. See `public/datasets/jrc_hela-2.yaml` for the
schema. Required fields: `name`, `display_name`, `voxel_size_nm`,
`layers`. Each layer needs `name`, `type` (`image` or `segmentation`),
and `source` (a Neuroglancer source URL — `zarr://`, `n5://`,
`precomputed://`, etc.).
