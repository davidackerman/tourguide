import {
  buildDescriptorFromInput,
  loadDescriptorFromYaml,
  descriptorToYaml,
  type PastedDatasetInput,
  type PastedLayerInput,
} from "./loader.js";
import type { DatasetDescriptor, LayerType } from "./descriptor.js";
import { detectSourceMetadata } from "./detect.js";
import { isFsAccessSupported, pickLocalFolder, buildLocalSourceUrl } from "./local_folder.js";

type LoaderResult = (descriptor: DatasetDescriptor) => void;

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
        <button class="modal-tab" data-tab="server" role="tab">Local server</button>
      </div>
      <div class="modal-body">
        <section class="modal-pane active" data-pane="form">
          <p class="hint">Point at data your browser can already reach (public S3, lab server, etc). Paste a source URL and tourguide will auto-detect voxel size and dataset center from its metadata.</p>
          <div class="form-row">
            <label>Name <input data-field="name" placeholder="my_dataset" /></label>
            <label>Voxel size (nm, auto-detected)
              <span class="voxel-row">
                <input type="number" step="any" data-field="vx" placeholder="x" />
                <input type="number" step="any" data-field="vy" placeholder="y" />
                <input type="number" step="any" data-field="vz" placeholder="z" />
              </span>
            </label>
          </div>
          <div class="form-row">
            <label>Initial position (nm, auto-detected from dataset center)
              <span class="voxel-row">
                <input type="number" step="any" data-field="px" placeholder="x" />
                <input type="number" step="any" data-field="py" placeholder="y" />
                <input type="number" step="any" data-field="pz" placeholder="z" />
              </span>
            </label>
          </div>
          <h3>Layers</h3>
          <div class="layers-list" data-layers></div>
          <button class="btn-secondary" data-add-layer>+ Add layer</button>
          <div class="detect-status" data-detect-status></div>
          <div class="form-actions">
            <button class="btn-primary" data-action="load-form">Load</button>
          </div>
        </section>
        <section class="modal-pane" data-pane="folder">
          <p class="hint">Pick a folder from your computer. Tourguide reads files directly via the File System Access API — nothing uploads anywhere, nothing leaves your machine. Works in Chrome / Edge / Brave (Chromium-based browsers).</p>
          <div class="folder-pick-row">
            <button class="btn-primary" data-action="pick-folder">Open folder…</button>
            <span class="folder-pick-status" data-folder-status></span>
          </div>
          <div class="folder-detected" data-folder-detected hidden></div>
          <p class="hint warn folder-unsupported" data-folder-unsupported hidden>This browser doesn't support the File System Access API. Use Chrome / Edge, or use the <strong>Local server</strong> tab.</p>
        </section>
        <section class="modal-pane" data-pane="yaml">
          <p class="hint">Paste a complete dataset descriptor in YAML.</p>
          <textarea class="yaml-input" rows="18" placeholder="name: my_dataset&#10;display_name: My Dataset&#10;voxel_size_nm: [4, 4, 4]&#10;layers:&#10;  - name: image&#10;    type: image&#10;    source: zarr://..."></textarea>
          <div class="form-actions">
            <button class="btn-primary" data-action="load-yaml">Load</button>
          </div>
        </section>
        <section class="modal-pane" data-pane="server">
          <p class="hint">For data on your laptop. Run this in the data directory:</p>
          <pre class="code-block">npx http-server --cors -p 8080</pre>
          <p class="hint">Then go to the <strong>Paste URLs</strong> tab and use sources like <code>zarr://http://localhost:8080/my-data.zarr/</code>.</p>
          <p class="hint warn">Permalinks for <code>localhost</code> URLs only work for you — to share, your data needs a public URL.</p>
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
      // Also auto-name the layer if left blank: derive from the URL's last path component.
      if (!nameInput.value) {
        const tail = sourceInput.value.replace(/\/+$/, "").split("/").pop() ?? "";
        if (tail) nameInput.value = tail.replace(/\.(zarr|n5)$/i, "");
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

  const submitYaml = (): void => {
    setError("");
    try {
      const text = overlay.querySelector<HTMLTextAreaElement>(".yaml-input")!.value;
      if (!text.trim()) throw new Error("Paste a descriptor");
      const descriptor = loadDescriptorFromYaml(text);
      onLoad(descriptor);
      close();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  overlay.querySelector("[data-action='load-form']")!.addEventListener("click", submitForm);
  overlay.querySelector("[data-action='load-yaml']")!.addEventListener("click", submitYaml);

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
      // Probe the picked folder for common formats and offer one-click loading.
      folderDetectedEl.hidden = false;
      folderDetectedEl.innerHTML = `
        <p class="hint">Trying to detect a dataset under <code>${escapeHtml(reg.name)}</code>…</p>
      `;
      const candidates: Array<{ kind: "zarr" | "n5" | "precomputed"; subpath: string }> = [];
      // Probe a few common entry-points: root and one level deep.
      for (const sub of ["", "data.zarr/", "data.n5/", "image/"]) {
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
        folderDetectedEl.innerHTML = `
          <p class="hint">✓ Detected ${f.kind} at <code>${escapeHtml(sub)}</code> — voxel ${f.meta.voxel_size_nm.join(" × ")} nm.</p>
          <button class="btn-primary" data-action="load-detected">Load this dataset</button>
        `;
        folderDetectedEl.querySelector("[data-action='load-detected']")!.addEventListener("click", () => {
          try {
            // Use the dtype-derived guess if detect returned one, otherwise
            // default to image. The user can still toggle in Paste URLs view.
            const guessed = (f.meta.guessedType ?? "image") as "image" | "segmentation";
            const baseName = f.subpath ? f.subpath.replace(/[/.]+$/, "") : guessed;
            const layers = [{
              name: baseName || guessed,
              type: guessed,
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

  document.body.appendChild(overlay);
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
