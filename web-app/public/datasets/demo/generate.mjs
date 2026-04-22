// Generate small synthetic CSVs so the structured browser can be exercised
// end-to-end without committing real organelle data.
// Run with: node web-app/public/datasets/demo/generate.mjs

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

function writeCsv(name, count, volumeScale) {
  const rng = mulberry32(seedFor(name));
  const rows = [["object_id", "volume", "surface_area", "position_x", "position_y", "position_z"]];
  for (let i = 1; i <= count; i++) {
    const v = Math.round(volumeScale * (0.2 + 10 * Math.pow(rng(), 3)));
    const sa = Math.round(Math.pow(v, 2 / 3) * (5 + rng() * 2));
    const px = Math.round(rng() * 24000);
    const py = Math.round(rng() * 3200);
    const pz = Math.round(rng() * 16680);
    rows.push([i, v, sa, px, py, pz]);
  }
  const text = rows.map((r) => r.join(",")).join("\n") + "\n";
  writeFileSync(join(__dirname, `${name}.csv`), text);
  console.log(`wrote ${name}.csv: ${count} rows`);
}

function seedFor(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return h || 1;
}
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

writeCsv("mitochondria", 250, 2e8);
writeCsv("nucleus", 12, 5e10);
writeCsv("endoplasmic_reticulum", 40, 1e9);
