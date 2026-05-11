import {
  type Catalog,
  type CatalogEntry,
  type DatasetDescriptor,
  parseDescriptor,
} from "./descriptor.js";

export async function fetchCatalog(catalogUrl: string): Promise<Catalog> {
  const res = await fetch(catalogUrl);
  if (!res.ok) {
    throw new Error(`Failed to fetch catalog ${catalogUrl}: HTTP ${res.status}`);
  }
  const data = (await res.json()) as Catalog;
  if (!data || typeof data !== "object" || !Array.isArray(data.datasets)) {
    throw new Error("Catalog JSON missing datasets array");
  }
  return data;
}

export async function fetchDescriptor(
  entry: CatalogEntry,
  baseUrl: string,
): Promise<DatasetDescriptor> {
  const url = new URL(entry.url, baseUrl).toString();
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch descriptor ${url}: HTTP ${res.status}`);
  }
  const text = await res.text();
  return parseDescriptor(text);
}
