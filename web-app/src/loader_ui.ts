import {
  buildDescriptorFromInput,
  loadDescriptorFromYaml,
  descriptorToYaml,
  resolveDescriptorAgainstFolder,
  defaultLayerName,
  type PastedDatasetInput,
  type PastedLayerInput,
} from "./loader.js";
import type { DatasetDescriptor, LayerType } from "./descriptor.js";
import { detectSourceMetadata } from "./detect.js";
import { isFsAccessSupported, pickLocalFolder, buildLocalSourceUrl } from "./local_folder.js";
// urlSafeParse accepts both standard JSON and NG's URL-safe form
// (single-quoted strings, '_' as a separator) — what you get when you
// copy NG's URL hash directly. Plain JSON.parse can't handle the latter.
import { urlSafeParse } from "neuroglancer/unstable/util/json.js";

// Second arg is the (optional) Neuroglancer state to overlay after the
// descriptor loads — used by the "Paste NG state" tab so the recipient
// also gets the camera + selected segments the source URL captured.
type LoaderResult = (descriptor: DatasetDescriptor, ngState?: Record<string, unknown>) => void;

export function openLoaderDialog(onLoad: LoaderResult): void {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.innerHTML = `
    <div class="modal" role="dialog" aria-label="Load your data">
      <header class="modal-header">
        <h2>Load your data</h2>
        <button class="modal-close" aria-label="Close">×</button>
      </header>
      <div class="modal-tabs" role="tablist">
        <button class="modal-tab active" data-tab="form" role="tab">Paste URLs</button>
        <button class="modal-tab" data-tab="folder" role="tab">Local folder</button>
        <button class="modal-tab" data-tab="yaml" role="tab">YAML</button>
        <button class="modal-tab" data-tab="ngstate" role="tab">NG state</button>
      </div>
      <div class="modal-body">
        <section class="modal-pane active" data-pane="form">
          <p class="hint">Paste a source URL — tourguide auto-detects voxel size and dataset center from its OME-Zarr / N5 / precomputed metadata. Works for public S3, your lab server, or <code>localhost</code> via <code>npx http-server --cors -p 8080</code> in your data dir.</p>
          <div class="form-row">
            <label>Name <input data-field="name" placeholder="my_dataset" /></label>
          </div>
          <h3>Layers</h3>
          <div class="layers-list" data-layers></div>
          <button class="btn-secondary" data-add-layer>+ Add layer</button>
          <details class="loader-advanced">
            <summary>Override voxel size / position (auto-detected from metadata otherwise)</summary>
            <div class="form-row">
              <label>Voxel size (nm)
                <span class="voxel-row">
                  <input type="number" step="any" data-field="vx" placeholder="x" />
                  <input type="number" step="any" data-field="vy" placeholder="y" />
                  <input type="number" step="any" data-field="vz" placeholder="z" />
                </span>
              </label>
              <label>Initial position (nm)
                <span class="voxel-row">
                  <input type="number" step="any" data-field="px" placeholder="x" />
                  <input type="number" step="any" data-field="py" placeholder="y" />
                  <input type="number" step="any" data-field="pz" placeholder="z" />
                </span>
              </label>
            </div>
          </details>
          <div class="detect-status" data-detect-status></div>
          <div class="form-actions">
            <button class="btn-primary" data-action="load-form">Load</button>
          </div>
        </section>
        <section class="modal-pane" data-pane="folder">
          <p class="hint">Pick a folder from your computer. Tourguide reads files directly via the File System Access API — nothing uploads, nothing leaves your machine. Chromium-based browsers only (Chrome / Edge / Brave).</p>
          <div class="folder-pick-row">
            <button class="btn-primary" data-action="pick-folder">Open folder…</button>
            <span class="folder-pick-status" data-folder-status></span>
          </div>
          <div class="folder-detected" data-folder-detected hidden></div>
          <p class="hint warn folder-unsupported" data-folder-unsupported hidden>This browser doesn't support the File System Access API. Use Chrome / Edge, or run <code>npx http-server --cors -p 8080</code> in your data dir and use <strong>Paste URLs</strong>.</p>
        </section>
        <section class="modal-pane" data-pane="yaml">
          <p class="hint">Paste a YAML descriptor. Sources can be absolute (<code>zarr://https://…</code>, <code>precomputed://https://…</code>) or relative to picked local folders.</p>
          <p class="hint">Pointing at local data: declare a <code>folders:</code> block — Load prompts for each folder pick.</p>
          <pre class="code-block">name: jrc_c-elegans-bw-1
folders:
  recon: Pick recon-1/ inside the .zarr
layers:
  - { type: image,        source: recon/em/fibsem-int16 }
  - { type: segmentation, source: recon/labels/inference/segmentations/mito }</pre>
          <p class="hint"><code>voxel_size_nm</code> and layer <code>name</code> are auto-detected when omitted.</p>
          <textarea class="yaml-input" rows="14" placeholder="name: my_dataset&#10;layers:&#10;  - type: image&#10;    source: zarr://..."></textarea>
          <input type="file" data-yaml-file accept=".yaml,.yml,application/yaml,text/yaml,text/plain" hidden />
          <div class="form-actions" style="display:flex;gap:8px;align-items:center;">
            <button class="btn-secondary" data-action="pick-yaml-file" type="button">Open YAML file…</button>
            <button class="btn-primary" data-action="load-yaml">Load</button>
          </div>
        </section>
        <section class="modal-pane" data-pane="ngstate">
          <p class="hint">Paste any of:
            <strong>(a)</strong> a full Neuroglancer URL like <code>https://neuroglancer-demo.appspot.com/#!{...}</code> — the prefix gets stripped automatically;
            <strong>(b)</strong> just the <code>#!{...}</code> hash fragment;
            <strong>(c)</strong> the raw JSON from NG's <strong>{ } JSON</strong> panel.
            Tourguide extracts layers (adding the right <code>zarr://</code> / <code>precomputed://</code> scheme prefixes), auto-fills voxel size from the first source, and overlays the camera + selected segments after load. Optionally attach a CSV per segmentation layer to enable the structured browser.</p>
          <div class="form-row">
            <label>Name <input data-ngstate-name placeholder="my_dataset" /></label>
          </div>
          <textarea class="ngstate-input" rows="10" data-ngstate-input placeholder='https://neuroglancer-demo.appspot.com/#!{"layers":[...]}  — or paste just the JSON'></textarea>
          <div class="form-actions" style="display:flex;gap:8px;align-items:center;">
            <button class="btn-secondary" data-action="parse-ngstate" type="button">Parse layers</button>
            <span class="ngstate-status" data-ngstate-status></span>
          </div>
          <div class="ngstate-layers" data-ngstate-layers hidden></div>
          <div class="form-actions">
            <button class="btn-primary" data-action="load-ngstate">Load</button>
          </div>
        </section>
      </div>
      <div class="modal-error" data-error></div>
    </div>
  `;

  const errorBox = overlay.querySelector<HTMLDivElement>("[data-error]")!;
  const layersList = overlay.querySelector<HTMLDivElement>("[data-layers]")!;

  const close = (): void => overlay.remove();
  overlay.querySelector(".modal-close")!.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  overlay.querySelectorAll<HTMLButtonElement>(".modal-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      overlay.querySelectorAll(".modal-tab").forEach((t) => t.classList.remove("active"));
      overlay.querySelectorAll(".modal-pane").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const tabName = tab.dataset.tab!;
      overlay.querySelector(`[data-pane="${tabName}"]`)!.classList.add("active");
      errorBox.textContent = "";
    });
  });

  const detectStatus = overlay.querySelector<HTMLDivElement>("[data-detect-status]")!;
  const setDetectStatus = (msg: string, kind: "" | "ok" | "err" | "pending" = ""): void => {
    detectStatus.textContent = msg;
    detectStatus.className = `detect-status ${kind}`;
  };

  const vxEl = overlay.querySelector<HTMLInputElement>(`[data-field="vx"]`)!;
  const vyEl = overlay.querySelector<HTMLInputElement>(`[data-field="vy"]`)!;
  const vzEl = overlay.querySelector<HTMLInputElement>(`[data-field="vz"]`)!;
  const pxEl = overlay.querySelector<HTMLInputElement>(`[data-field="px"]`)!;
  const pyEl = overlay.querySelector<HTMLInputElement>(`[data-field="py"]`)!;
  const pzEl = overlay.querySelector<HTMLInputElement>(`[data-field="pz"]`)!;

  const tryAutoDetect = async (sourceUrl: string): Promise<void> => {
    const trimmed = sourceUrl.trim();
    if (!trimmed) return;
    setDetectStatus(`Fetching metadata from ${trimmed.slice(0, 80)}…`, "pending");
    try {
      const meta = await detectSourceMetadata(trimmed);
      if (!vxEl.value) vxEl.value = String(meta.voxel_size_nm[0]);
      if (!vyEl.value) vyEl.value = String(meta.voxel_size_nm[1]);
      if (!vzEl.value) vzEl.value = String(meta.voxel_size_nm[2]);
      if (meta.center_nm) {
        if (!pxEl.value) pxEl.value = String(Math.round(meta.center_nm[0]));
        if (!pyEl.value) pyEl.value = String(Math.round(meta.center_nm[1]));
        if (!pzEl.value) pzEl.value = String(Math.round(meta.center_nm[2]));
      }
      const pos = meta.center_nm ? `, center ${meta.center_nm.map((n) => Math.round(n)).join(", ")} nm` : "";
      setDetectStatus(
        `✓ Detected voxel ${meta.voxel_size_nm.join(" × ")} nm${pos}  (${meta.via})`,
        "ok",
      );
    } catch (err) {
      setDetectStatus(
        `✗ Auto-detect failed: ${(err as Error).message}. You can still enter values manually.`,
        "err",
      );
    }
  };

  const addLayerRow = (): void => {
    const row = document.createElement("div");
    row.className = "layer-row";
    row.innerHTML = `
      <input data-l="name" placeholder="layer name" />
      <select data-l="type">
        <option value="image">image</option>
        <option value="segmentation">segmentation</option>
      </select>
      <input data-l="source" placeholder="zarr://… or n5://… or precomputed://…" />
      <input data-l="organelle_class" placeholder="organelle class (optional)" />
      <input data-l="csv" placeholder="CSV URL (optional)" />
      <button class="btn-remove" aria-label="Remove">×</button>
    `;
    row.querySelector(".btn-remove")!.addEventListener("click", () => row.remove());
    const sourceInput = row.querySelector<HTMLInputElement>(`[data-l="source"]`)!;
    const nameInput = row.querySelector<HTMLInputElement>(`[data-l="name"]`)!;
    sourceInput.addEventListener("blur", () => {
      void tryAutoDetect(sourceInput.value);
      // Auto-name from URL when the name field is blank — same helper
      // buildDescriptorFromInput uses, so behavior matches whether the
      // user blurs the field or not.
      if (!nameInput.value && sourceInput.value.trim()) {
        nameInput.value = defaultLayerName(sourceInput.value);
      }
    });
    layersList.appendChild(row);
  };

  overlay.querySelector("[data-add-layer]")!.addEventListener("click", () => addLayerRow());
  addLayerRow();

  const setError = (msg: string): void => {
    errorBox.textContent = msg;
  };

  const submitForm = (): void => {
    setError("");
    try {
      const get = (field: string): string => {
        const el = overlay.querySelector<HTMLInputElement>(`[data-field="${field}"]`);
        return el?.value.trim() ?? "";
      };
      const num = (field: string): number => {
        const v = get(field);
        if (v === "") return NaN;
        return Number(v);
      };
      const name = get("name");
      if (!name) throw new Error("Name is required");
      const voxel: [number, number, number] = [num("vx"), num("vy"), num("vz")];
      if (voxel.some((v) => !Number.isFinite(v) || v <= 0)) {
        throw new Error("Voxel size must be 3 positive numbers");
      }
      const px = num("px"), py = num("py"), pz = num("pz");
      const initialPosition: [number, number, number] | undefined =
        [px, py, pz].every((v) => Number.isFinite(v))
          ? [px, py, pz]
          : undefined;

      const layers: PastedLayerInput[] = [];
      layersList.querySelectorAll<HTMLDivElement>(".layer-row").forEach((row) => {
        const lget = (k: string): string => {
          const el = row.querySelector<HTMLInputElement | HTMLSelectElement>(`[data-l="${k}"]`);
          return el?.value.trim() ?? "";
        };
        const layer: PastedLayerInput = {
          name: lget("name"),
          type: lget("type") as LayerType,
          source: lget("source"),
          organelle_class: lget("organelle_class") || undefined,
          csv: lget("csv") || undefined,
        };
        if (layer.name && layer.source) layers.push(layer);
      });
      if (layers.length === 0) throw new Error("Add at least one layer");

      const input: PastedDatasetInput = {
        name,
        voxel_size_nm: voxel,
        initial_position: initialPosition,
        layers,
      };
      const descriptor = buildDescriptorFromInput(input);
      onLoad(descriptor);
      console.info("Loaded user descriptor:\n" + descriptorToYaml(descriptor));
      close();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const submitYaml = async (): Promise<void> => {
    setError("");
    try {
      const text = overlay.querySelector<HTMLTextAreaElement>(".yaml-input")!.value;
      if (!text.trim()) throw new Error("Paste a descriptor");
      const descriptor = loadDescriptorFromYaml(text);
      let resolved = descriptor;
      // If the descriptor declares a `folders:` block, prompt the user
      // to pick each folder in turn (with the YAML's hint as a label),
      // register each, and resolve relative sources against the
      // resulting base URLs. Any source already absolute passes through.
      if (descriptor.folders && Object.keys(descriptor.folders).length > 0) {
        const baseMap: Record<string, string> = {};
        for (const [alias, hint] of Object.entries(descriptor.folders)) {
          const friendly = `For folder "${alias}": ${hint || "pick the folder containing this layer's data"}.\n\nClick OK to open the folder picker.`;
          if (!confirm(friendly)) throw new Error(`Folder pick for '${alias}' cancelled`);
          const reg = await pickLocalFolder();
          baseMap[alias] = reg.baseUrl;
        }
        resolved = resolveDescriptorAgainstFolder(descriptor, baseMap);
      }
      resolved = await autofillVoxelFromFirstSource(resolved);
      onLoad(resolved);
      close();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  // Read voxel size + center from the first zarr/n5/precomputed layer's
  // metadata when the descriptor either omitted voxel_size_nm or left
  // it at the validateDescriptor placeholder [1,1,1]. Lets users drop
  // 'voxel_size_nm' from their YAML entirely for OME-Zarr datasets.
  const autofillVoxelFromFirstSource = async (
    d: DatasetDescriptor,
  ): Promise<DatasetDescriptor> => {
    const v = d.voxel_size_nm;
    const isPlaceholder = v && v[0] === 1 && v[1] === 1 && v[2] === 1;
    if (!isPlaceholder) return d;
    const first = d.layers[0];
    if (!first) return d;
    try {
      // Multi-source layers: probe the first source for metadata (the
      // others are usually mesh/skeleton attachments without volume).
      const probeUrl = Array.isArray(first.source) ? first.source[0] : first.source;
      const meta = await detectSourceMetadata(probeUrl);
      return {
        ...d,
        voxel_size_nm: meta.voxel_size_nm,
        initial_position: d.initial_position ?? meta.center_nm,
      };
    } catch (err) {
      console.warn("[loader] auto-detect voxel size failed; keeping [1,1,1]:", (err as Error).message);
      return d;
    }
  };

  overlay.querySelector("[data-action='load-form']")!.addEventListener("click", submitForm);
  overlay.querySelector("[data-action='load-yaml']")!.addEventListener("click", () => void submitYaml());
  // Open-file → read text → drop into the textarea so the user can
  // visually confirm before clicking Load. Means a YAML stored next to
  // the data (or in your home dir as a saved descriptor) loads with
  // two clicks: pick file, hit Load.
  const yamlFileInput = overlay.querySelector<HTMLInputElement>("[data-yaml-file]")!;
  overlay.querySelector("[data-action='pick-yaml-file']")!.addEventListener("click", () => yamlFileInput.click());
  yamlFileInput.addEventListener("change", async () => {
    const f = yamlFileInput.files?.[0];
    if (!f) return;
    try {
      const text = await f.text();
      overlay.querySelector<HTMLTextAreaElement>(".yaml-input")!.value = text;
    } catch (err) {
      setError((err as Error).message);
    }
  });

  // ---- Local folder tab ----
  const folderUnsupported = overlay.querySelector<HTMLElement>("[data-folder-unsupported]")!;
  const folderStatusEl = overlay.querySelector<HTMLSpanElement>("[data-folder-status]")!;
  const folderDetectedEl = overlay.querySelector<HTMLDivElement>("[data-folder-detected]")!;
  const pickBtn = overlay.querySelector<HTMLButtonElement>("[data-action='pick-folder']")!;
  if (!isFsAccessSupported()) {
    folderUnsupported.hidden = false;
    pickBtn.disabled = true;
  }
  pickBtn.addEventListener("click", async () => {
    pickBtn.disabled = true;
    folderStatusEl.textContent = "Waiting for folder pick…";
    folderStatusEl.className = "folder-pick-status pending";
    folderDetectedEl.hidden = true;
    try {
      const reg = await pickLocalFolder();
      folderStatusEl.textContent = `Picked: ${reg.name}`;
      folderStatusEl.className = "folder-pick-status ok";
      folderDetectedEl.hidden = false;
      // First, look for a tourguide.yaml/.yml dropped at the root of the
      // picked folder. If found, parse it and rewrite relative sources
      // against the folder's base URL — that's a one-click load with full
      // user-authored layer metadata (csv links, organelle classes, etc.)
      // beating any auto-detect heuristic.
      const yamlNames = ["tourguide.yaml", "tourguide.yml", "descriptor.yaml", "descriptor.yml"];
      let resolvedDescriptor: DatasetDescriptor | null = null;
      for (const fname of yamlNames) {
        try {
          const res = await fetch(reg.baseUrl + fname);
          if (!res.ok) continue;
          const yamlText = await res.text();
          const parsed = loadDescriptorFromYaml(yamlText);
          resolvedDescriptor = resolveDescriptorAgainstFolder(parsed, reg.baseUrl);
          folderDetectedEl.innerHTML = `
            <p class="hint">✓ Found <code>${escapeHtml(fname)}</code> in <code>${escapeHtml(reg.name)}</code> — relative sources resolved to this folder.</p>
            <button class="btn-primary" data-action="load-yaml-from-folder">Load this dataset</button>
          `;
          folderDetectedEl.querySelector("[data-action='load-yaml-from-folder']")!.addEventListener("click", () => {
            try {
              onLoad(resolvedDescriptor!);
              close();
            } catch (err) {
              setError((err as Error).message);
            }
          });
          break;
        } catch {
          /* keep looking; fall through to auto-detect */
        }
      }
      if (resolvedDescriptor) return;
      // No tourguide.yaml — fall back to probing common entry points.
      folderDetectedEl.innerHTML = `
        <p class="hint">No <code>tourguide.yaml</code> in this folder. Trying to auto-detect a dataset under <code>${escapeHtml(reg.name)}</code>…</p>
      `;
      const candidates: Array<{ kind: "zarr" | "n5" | "precomputed"; subpath: string }> = [];
      // Probe a few common entry-points: root, one level deep, and a few
      // typical precomputed mesh layouts. First hit wins.
      const subpaths = [
        "",
        "data.zarr/", "data.n5/", "image/",
        "mesh/", "meshes/", "segmentation/", "segmentations/",
      ];
      for (const sub of subpaths) {
        for (const kind of ["zarr", "n5", "precomputed"] as const) {
          candidates.push({ kind, subpath: sub });
        }
      }
      const found: Array<{ kind: string; subpath: string; meta: { voxel_size_nm: [number, number, number]; via: string; center_nm?: [number, number, number]; guessedType?: "image" | "segmentation" } }> = [];
      for (const c of candidates) {
        try {
          const url = buildLocalSourceUrl(reg, c.kind, c.subpath);
          const meta = await detectSourceMetadata(url);
          found.push({ kind: c.kind, subpath: c.subpath, meta });
          break; // first hit wins
        } catch {
          // try next
        }
      }
      if (found.length === 0) {
        folderDetectedEl.innerHTML = `
          <p class="hint">No top-level dataset detected. Switch to the <strong>Paste URLs</strong> tab and use a source URL like:</p>
          <pre class="code-block">zarr://${reg.baseUrl}path/to/your.zarr/</pre>
        `;
      } else {
        const f = found[0];
        const sub = f.subpath ? f.subpath : "(root)";
        const url = buildLocalSourceUrl(reg, f.kind as "zarr" | "n5" | "precomputed", f.subpath);
        const guessed = (f.meta.guessedType ?? "image") as "image" | "segmentation";
        // Show the dtype-based guess but let the user override before
        // loading — auto-detection is wrong for label volumes whose
        // dtype is uint8 (e.g. paintera intermediate exports), and the
        // type can't be changed once the layer is in NG.
        folderDetectedEl.innerHTML = `
          <p class="hint" style="margin:0 0 8px;">✓ ${f.kind} at <code>${escapeHtml(sub)}</code> — voxel ${f.meta.voxel_size_nm.join(" × ")} nm.</p>
          <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
            <label style="display:flex;gap:6px;align-items:center;">
              <span>Load as:</span>
              <select data-load-type>
                <option value="image"${guessed === "image" ? " selected" : ""}>image (intensity)</option>
                <option value="segmentation"${guessed === "segmentation" ? " selected" : ""}>segmentation (labels)</option>
              </select>
            </label>
            <button class="btn-primary" data-action="load-detected">Load</button>
            <span class="hint" style="opacity:0.6;">(guessed: ${guessed})</span>
          </div>
        `;
        folderDetectedEl.querySelector("[data-action='load-detected']")!.addEventListener("click", () => {
          try {
            const sel = folderDetectedEl.querySelector<HTMLSelectElement>("[data-load-type]");
            const chosenType = (sel?.value as "image" | "segmentation") ?? guessed;
            const layers = [{
              name: defaultLayerName(url, chosenType),
              type: chosenType,
              source: url,
            }];
            const descriptor = buildDescriptorFromInput({
              name: reg.name,
              voxel_size_nm: f.meta.voxel_size_nm,
              initial_position: f.meta.center_nm,
              layers,
            });
            onLoad(descriptor);
            close();
          } catch (err) {
            setError((err as Error).message);
          }
        });
      }
    } catch (err) {
      folderStatusEl.textContent = (err as Error).message;
      folderStatusEl.className = "folder-pick-status err";
    } finally {
      pickBtn.disabled = false;
    }
  });

  // ---- NG state tab ----
  const ngstateInput = overlay.querySelector<HTMLTextAreaElement>("[data-ngstate-input]")!;
  const ngstateNameInput = overlay.querySelector<HTMLInputElement>("[data-ngstate-name]")!;
  const ngstateLayersHost = overlay.querySelector<HTMLDivElement>("[data-ngstate-layers]")!;
  const ngstateStatusEl = overlay.querySelector<HTMLSpanElement>("[data-ngstate-status]")!;
  const setNgstateStatus = (msg: string, kind: "" | "ok" | "err" = ""): void => {
    ngstateStatusEl.textContent = msg;
    ngstateStatusEl.className = `ngstate-status ${kind}`;
  };

  // Pull the layer specs out of the pasted NG state. NG accepts both
  // array and object forms; reduce both to {name, type, source} we can
  // feed buildDescriptorFromInput. Source can itself be a string, an
  // array, or {url} / [{url}] — flatten to string|string[].
  const extractLayersFromNgState = (
    ng: Record<string, unknown>,
  ): PastedLayerInput[] => {
    const raw = ng.layers;
    if (raw === undefined) throw new Error("NG state is missing 'layers'");
    let entries: Array<Record<string, unknown>>;
    if (Array.isArray(raw)) {
      entries = raw.filter((l): l is Record<string, unknown> => !!l && typeof l === "object");
    } else if (typeof raw === "object" && raw !== null) {
      entries = Object.entries(raw as Record<string, unknown>).map(([name, spec]) => {
        if (typeof spec === "string") return { name, source: spec };
        if (spec && typeof spec === "object") return { name, ...(spec as Record<string, unknown>) };
        return { name };
      });
    } else {
      throw new Error("NG state 'layers' must be an array or object");
    }
    // NG infers the data-source kind from URL shape (a path containing
    // .zarr/ → zarr datasource), so its state JSON often omits the
    // 'zarr://' prefix. Tourguide downstream code (analysis_ui, etc.)
    // identifies zarr layers by checking for that prefix, so we re-add
    // it here when the URL is clearly a zarr path. Keeps existing
    // schemes intact (precomputed://, n5://, gs://, etc.).
    const ensureScheme = (url: string): string => {
      if (/^[a-z][a-z0-9+]*:\/\//i.test(url)) return url; // already has scheme
      // Heuristic: '.zarr/' or trailing '.zarr' → zarr datasource.
      // 'precomputed' segment in the path → precomputed source.
      // Otherwise leave bare (n5 etc. is harder to detect from URL alone).
      if (/\.zarr(\/|$)/i.test(url)) return `zarr://${url}`;
      if (/\/(precomputed|neuroglancer\/(mesh|skeleton))\b/i.test(url)) return `precomputed://${url}`;
      if (/\.n5(\/|$)/i.test(url)) return `n5://${url}`;
      return url;
    };
    const flattenSource = (s: unknown): string | string[] | null => {
      const one = (x: unknown): string | null => {
        if (typeof x === "string") return ensureScheme(x);
        if (x && typeof x === "object" && typeof (x as { url?: unknown }).url === "string") {
          return ensureScheme((x as { url: string }).url);
        }
        return null;
      };
      if (Array.isArray(s)) {
        const out = s.map(one).filter((v): v is string => !!v);
        if (out.length === 0) return null;
        return out.length === 1 ? out[0] : out;
      }
      return one(s);
    };
    const out: PastedLayerInput[] = [];
    for (const entry of entries) {
      const name = typeof entry.name === "string" ? entry.name : "";
      const type = entry.type === "segmentation" ? "segmentation" : "image";
      const src = flattenSource(entry.source);
      if (!src) continue;
      // Keep multi-source layers as arrays — preserves NG's
      // zarr-volume + precomputed-mesh + precomputed-skeleton bundle.
      // Analysis tools that need ONE zarr URL pick the right one via
      // pickZarrEntry; the viewer renders all sources together.
      const nameSource = Array.isArray(src) ? src[0] : src;
      out.push({
        name: name || defaultLayerName(nameSource, type),
        type,
        source: src,
      });
    }
    if (out.length === 0) {
      throw new Error("No layers with a usable source found in the NG state");
    }
    return out;
  };

  const renderNgstateLayerRows = (layers: PastedLayerInput[]): void => {
    ngstateLayersHost.hidden = false;
    ngstateLayersHost.innerHTML = `
      <h3 style="margin:8px 0 4px;">Detected layers</h3>
      <p class="hint">Source URLs are taken straight from the NG state; the CSV / organelle-class fields below are tourguide-only and stay blank unless you fill them in.</p>
      <div class="layers-list" data-ngstate-layer-rows></div>
    `;
    const rowsHost = ngstateLayersHost.querySelector<HTMLDivElement>("[data-ngstate-layer-rows]")!;
    layers.forEach((l, i) => {
      const row = document.createElement("div");
      row.className = "layer-row";
      row.dataset.idx = String(i);
      row.innerHTML = `
        <input data-l="name" value="${escapeAttr(l.name)}" />
        <select data-l="type">
          <option value="image"${l.type === "image" ? " selected" : ""}>image</option>
          <option value="segmentation"${l.type === "segmentation" ? " selected" : ""}>segmentation</option>
        </select>
        <input data-l="source" value="${escapeAttr(Array.isArray(l.source) ? l.source.join(", ") : l.source)}" readonly title="Source comes from the NG state" />
        <input data-l="organelle_class" placeholder="organelle class (optional)" />
        <input data-l="csv" placeholder="CSV URL (optional)" />
      `;
      rowsHost.appendChild(row);
    });
  };

  const parseNgstate = (): { ng: Record<string, unknown>; layers: PastedLayerInput[] } | null => {
    setError("");
    setNgstateStatus("");
    const raw = ngstateInput.value.trim();
    if (!raw) {
      setError("Paste an NG state JSON first");
      return null;
    }
    // Accept whatever shape the user pastes:
    //   - The bare `{...}` from NG's "{ } JSON" panel (standard JSON)
    //   - The `#!{...}` form from NG's URL hash (URL-encoded JSON)
    //   - The `#!+{...}` form (URL-safe encoding: single quotes, `_` for `,`)
    //   - A full `https://.../#!...` URL pasted in
    let cleaned = raw;
    const hashIdx = cleaned.indexOf("#!");
    if (hashIdx >= 0) cleaned = cleaned.slice(hashIdx + 2);
    if (cleaned.startsWith("+")) cleaned = cleaned.slice(1); // legacy `#!+` form
    if (!cleaned.startsWith("{")) {
      // URL-encoded variants — `%7B` etc. — survive a copy from the
      // address bar in some browsers.
      try {
        cleaned = decodeURIComponent(cleaned);
      } catch {
        /* fall through; urlSafeParse will give the real error */
      }
    }
    let ng: Record<string, unknown>;
    try {
      // urlSafeParse handles both `{"layers":...}` and `{'layers':...}`;
      // JSON.parse rejects the latter, which is what NG actually puts in
      // the URL by default.
      ng = urlSafeParse(cleaned);
    } catch (err) {
      setError(`Invalid JSON: ${(err as Error).message}`);
      return null;
    }
    if (!ng || typeof ng !== "object") {
      setError("NG state must be a JSON object");
      return null;
    }
    let layers: PastedLayerInput[];
    try {
      layers = extractLayersFromNgState(ng);
    } catch (err) {
      setError((err as Error).message);
      return null;
    }
    if (!ngstateNameInput.value) {
      // Default the dataset name to the first layer's URL tail, matching
      // what the YAML loader does when name is omitted.
      const firstSource = Array.isArray(layers[0].source) ? layers[0].source[0] : layers[0].source;
      ngstateNameInput.value = defaultLayerName(firstSource, "ngstate");
    }
    setNgstateStatus(`✓ Found ${layers.length} layer(s)`, "ok");
    return { ng, layers };
  };

  overlay.querySelector("[data-action='parse-ngstate']")!.addEventListener("click", () => {
    const parsed = parseNgstate();
    if (parsed) renderNgstateLayerRows(parsed.layers);
  });

  // Re-parse on textarea blur so the layer rows refresh after edits, but
  // only if the user has already triggered an initial parse (otherwise
  // a paste-then-tab away would surprise them with an inline error).
  ngstateInput.addEventListener("blur", () => {
    if (!ngstateLayersHost.hidden) {
      const parsed = parseNgstate();
      if (parsed) renderNgstateLayerRows(parsed.layers);
    }
  });

  overlay.querySelector("[data-action='load-ngstate']")!.addEventListener("click", async () => {
    setError("");
    const parsed = parseNgstate();
    if (!parsed) return;
    // Pull any CSV / organelle-class overrides the user typed into the
    // detected-layer rows. If the user never clicked Parse, fall back to
    // the layers extracted just now (no overrides available).
    let layers = parsed.layers;
    const rows = ngstateLayersHost.querySelectorAll<HTMLDivElement>(".layer-row");
    if (rows.length === layers.length) {
      layers = Array.from(rows).map((row, i) => {
        const lget = (k: string): string => {
          const el = row.querySelector<HTMLInputElement | HTMLSelectElement>(`[data-l="${k}"]`);
          return el?.value.trim() ?? "";
        };
        return {
          name: lget("name") || layers[i].name,
          type: (lget("type") as LayerType) || layers[i].type,
          source: layers[i].source,
          organelle_class: lget("organelle_class") || undefined,
          csv: lget("csv") || undefined,
        };
      });
    }
    const name = ngstateNameInput.value.trim() || "ngstate_paste";
    try {
      // Build a placeholder descriptor first; voxel size is auto-filled
      // below the same way the YAML loader does it.
      let descriptor = buildDescriptorFromInput({
        name,
        voxel_size_nm: [1, 1, 1], // placeholder, autofill replaces below
        layers,
      });
      descriptor = await autofillVoxelFromFirstSource(descriptor);
      onLoad(descriptor, parsed.ng);
      close();
    } catch (err) {
      setError((err as Error).message);
    }
  });

  document.body.appendChild(overlay);
}

function escapeAttr(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
