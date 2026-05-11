import { defineConfig } from "vite";

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
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
  preview: {
    port: 4173,
    host: true,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
