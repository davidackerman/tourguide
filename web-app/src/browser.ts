import type { DatasetDB, IngestedTable, QueryResult } from "./db.js";
import { runQuery } from "./db.js";
import type { Viewer } from "./viewer.js";

const NAVIGABLE_NUMERIC_COLUMNS = ["volume", "surface_area"];

export interface BrowserContext {
  db: DatasetDB;
  viewer: Viewer;
}

interface SortState {
  column: string;
  direction: "asc" | "desc";
}

const ROWS_PER_PAGE = 50;

export function renderStructuredBrowser(container: HTMLElement, ctx: BrowserContext): void {
  container.innerHTML = "";
  if (ctx.db.tables.length === 0) {
    const empty = document.createElement("p");
    empty.className = "placeholder";
    empty.textContent = "No organelle metadata in this dataset (no layers had a CSV).";
    container.appendChild(empty);
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "browser";
  wrap.innerHTML = `
    <div class="browser-header">
      <label class="browser-class-picker">
        <span>Class</span>
        <select data-class></select>
      </label>
      <span class="browser-count" data-count></span>
    </div>
    <div class="browser-table-wrap">
      <table class="browser-table">
        <thead><tr data-thead></tr></thead>
        <tbody data-tbody></tbody>
      </table>
    </div>
    <div class="browser-footer">
      <span class="hint" data-page-info></span>
      <div class="browser-pager">
        <button class="btn-secondary" data-prev>‹ Prev</button>
        <button class="btn-secondary" data-next>Next ›</button>
      </div>
    </div>
  `;
  container.appendChild(wrap);

  const select = wrap.querySelector<HTMLSelectElement>("[data-class]")!;
  const countEl = wrap.querySelector<HTMLSpanElement>("[data-count]")!;
  const thead = wrap.querySelector<HTMLTableRowElement>("[data-thead]")!;
  const tbody = wrap.querySelector<HTMLTableSectionElement>("[data-tbody]")!;
  const pageInfo = wrap.querySelector<HTMLSpanElement>("[data-page-info]")!;
  const prevBtn = wrap.querySelector<HTMLButtonElement>("[data-prev]")!;
  const nextBtn = wrap.querySelector<HTMLButtonElement>("[data-next]")!;

  ctx.db.tables.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t.table_name;
    opt.textContent = `${t.organelle_class} (${t.row_count.toLocaleString()})`;
    select.appendChild(opt);
  });

  let currentTable: IngestedTable = ctx.db.tables[0];
  let sort: SortState = { column: "volume", direction: "desc" };
  let page = 0;

  const navigableColumns = (table: IngestedTable): string[] =>
    table.columns.filter((c) => NAVIGABLE_NUMERIC_COLUMNS.includes(c) || c === "object_id");

  const pickInitialSort = (table: IngestedTable): SortState => {
    if (table.columns.includes("volume")) return { column: "volume", direction: "desc" };
    if (table.columns.includes("surface_area")) return { column: "surface_area", direction: "desc" };
    return { column: "object_id", direction: "asc" };
  };

  const buildSql = (): string => {
    const colList = currentTable.columns.map((c) => `"${c}"`).join(", ");
    const dir = sort.direction.toUpperCase();
    return `SELECT ${colList} FROM "${currentTable.table_name}" ORDER BY "${sort.column}" ${dir} LIMIT ${ROWS_PER_PAGE} OFFSET ${page * ROWS_PER_PAGE};`;
  };

  const renderHeader = (): void => {
    thead.innerHTML = "";
    for (const col of currentTable.columns) {
      const th = document.createElement("th");
      th.textContent = col;
      th.dataset.col = col;
      if (col === sort.column) {
        th.classList.add("sorted");
        th.textContent = `${col} ${sort.direction === "desc" ? "↓" : "↑"}`;
      }
      th.addEventListener("click", () => {
        if (sort.column === col) {
          sort.direction = sort.direction === "desc" ? "asc" : "desc";
        } else {
          sort = { column: col, direction: "desc" };
        }
        page = 0;
        renderHeader();
        renderRows();
      });
      thead.appendChild(th);
    }
  };

  const renderRows = (): void => {
    let result: QueryResult;
    try {
      result = runQuery(ctx.db.db, buildSql());
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="${currentTable.columns.length}">Query error: ${(err as Error).message}</td></tr>`;
      return;
    }
    tbody.innerHTML = "";
    if (result.rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="${currentTable.columns.length}">no rows</td></tr>`;
    }
    for (const row of result.rows) {
      const tr = document.createElement("tr");
      tr.className = "row-clickable";
      currentTable.columns.forEach((_col, i) => {
        const td = document.createElement("td");
        const v = row[i];
        td.textContent = formatCell(v);
        if (v !== null && typeof v === "number") td.classList.add("num");
        tr.appendChild(td);
      });
      tr.addEventListener("click", () => {
        const rowObj = makeRowObject(currentTable.columns, row);
        flyFromRow(ctx.viewer, currentTable, rowObj);
      });
      tbody.appendChild(tr);
    }
    countEl.textContent = `${currentTable.row_count.toLocaleString()} rows`;
    const start = page * ROWS_PER_PAGE + 1;
    const end = Math.min((page + 1) * ROWS_PER_PAGE, currentTable.row_count);
    pageInfo.textContent = `${start.toLocaleString()}–${end.toLocaleString()} of ${currentTable.row_count.toLocaleString()}`;
    prevBtn.disabled = page === 0;
    nextBtn.disabled = end >= currentTable.row_count;
  };

  const switchTable = (tableName: string): void => {
    const t = ctx.db.tables.find((x) => x.table_name === tableName);
    if (!t) return;
    currentTable = t;
    sort = pickInitialSort(t);
    page = 0;
    renderHeader();
    renderRows();
    void navigableColumns;
  };

  select.addEventListener("change", () => switchTable(select.value));
  prevBtn.addEventListener("click", () => {
    if (page > 0) {
      page--;
      renderRows();
    }
  });
  nextBtn.addEventListener("click", () => {
    page++;
    renderRows();
  });

  switchTable(currentTable.table_name);
}

function makeRowObject(columns: string[], row: unknown[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  columns.forEach((c, i) => (out[c] = row[i]));
  return out;
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") {
    if (Number.isInteger(v) && Math.abs(v) < 1e7) return v.toLocaleString();
    if (Math.abs(v) >= 1e6 || (Math.abs(v) > 0 && Math.abs(v) < 1e-3)) return v.toExponential(2);
    return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  return String(v);
}

function flyFromRow(viewer: Viewer, table: IngestedTable, row: Record<string, unknown>): void {
  const px = row.position_x;
  const py = row.position_y;
  const pz = row.position_z;
  const id = row.object_id;
  if (typeof px === "number" && typeof py === "number" && typeof pz === "number") {
    viewer.flyTo([px, py, pz], id !== undefined ? String(id) : undefined, table.layer_name);
  } else {
    console.warn(`Row missing position columns; cannot fly. Row:`, row);
  }
}
