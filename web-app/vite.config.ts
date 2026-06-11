import { defineConfig, type PluginOption } from "vite";

// Content-hashed build assets never change for a given URL, so let the browser
// cache them forever (and reuse its compiled-code cache). vite preview
// otherwise sends `no-cache`, forcing every freshly-opened workspace tab to
// re-download + re-parse ~9 MB of Neuroglancer — the bulk of new-tab latency.
// index.html stays uncached so a rebuild's new hashes are picked up.
function immutableAssets(): PluginOption {
  return {
    name: "immutable-assets",
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.startsWith("/assets/")) {
          res.setHeader("Cache-Control", "public, max-age=31536000, immutable");
        }
        next();
      });
    },
  };
}

// The bundled JS Neuroglancer needs cross-origin isolation (COOP/COEP) for
// SharedArrayBuffer — but that isolation BLOCKS embedding a cross-origin
// Python-NG viewer in an iframe. In embedded mode (?ngViewer=…) we don't use
// the bundled viewer, so serve that page WITHOUT COEP so the iframe can load.
function crossOriginIsolationExceptEmbedded(): PluginOption {
  const apply = (server: { middlewares: { use: (fn: (req: any, res: any, next: () => void) => void) => void } }) => {
    server.middlewares.use((req, res, next) => {
      if (!(req.url && req.url.includes("ngViewer"))) {
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "credentialless");
      }
      next();
    });
  };
  return { name: "coi-except-embedded", configureServer: apply, configurePreviewServer: apply };
}

// Neuroglancer uses Node-style "exports" conditions to gate which datasources,
// kvstores, and layers get bundled. Enabling everything we care about for
// CellMap-style data: zarr/n5/precomputed sources over s3/http with the usual
// layer types. Add more here if a user needs e.g. dvid or graphene.
const NG_CONDITIONS = [
  "neuroglancer/datasource/zarr:enabled",
  "neuroglancer/datasource/n5:enabled",
  "neuroglancer/datasource/precomputed:enabled",
  "neuroglancer/datasource/render:enabled",
  "neuroglancer/datasource/nifti:enabled",
  "neuroglancer/datasource/obj:enabled",
  "neuroglancer/datasource/vtk:enabled",
  "neuroglancer/datasource/deepzoom:enabled",
  "neuroglancer/datasource/dvid:enabled",
  "neuroglancer/kvstore/http:enabled",
  "neuroglancer/kvstore/s3:enabled",
  "neuroglancer/kvstore/gcs:enabled",
  "neuroglancer/kvstore/gzip:enabled",
  "neuroglancer/kvstore/byte_range:enabled",
  "neuroglancer/kvstore/zip:enabled",
  "neuroglancer/kvstore/ocdbt:enabled",
  "neuroglancer/kvstore/icechunk:enabled",
  "neuroglancer/layer/image:enabled",
  "neuroglancer/layer/segmentation:enabled",
  "neuroglancer/layer/annotation:enabled",
  "neuroglancer/layer/single_mesh:enabled",
];

export default defineConfig({
  base: "./",
  plugins: [immutableAssets(), crossOriginIsolationExceptEmbedded()],
  resolve: {
    conditions: NG_CONDITIONS,
  },
  optimizeDeps: {
    // Pre-bundle Neuroglancer's deeply-nested ESM tree so dev startup is fast.
    include: ["neuroglancer"],
  },
  worker: {
    format: "es",
  },
  server: {
    port: 5173,
    host: true,
    // COOP/COEP are applied conditionally by crossOriginIsolationExceptEmbedded
    // (skipped for the ?ngViewer embed page so its cross-origin iframe loads).
  },
  preview: {
    port: 4173,
    host: true,
    // Allow access via the machine's LAN IP/hostname, not just localhost, so
    // others on the same network can open the workspace. Vite preview
    // otherwise rejects unknown Host headers ("host not allowed").
    allowedHosts: true,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
