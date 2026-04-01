import { Storage } from "@google-cloud/storage";
import type { GetFilesOptions } from "@google-cloud/storage";
import { gcsAllowedBlobPrefixes } from "@/lib/gcsPaths";

let storage: Storage | null = null;

function getStorage(): Storage {
  if (!storage) {
    storage = new Storage({
      projectId: process.env.GCP_PROJECT_ID,
    });
  }
  return storage;
}

const BUCKET_NAME = process.env.GCS_BUCKET_NAME || "cv-urban-accessibility-bucket";

/** Capture run folder segment, e.g. 20260329T023321Z or 20260329T023321Z-BAGDAT-30m */
const CAPTURE_RUN_FOLDER_RE = /^\d{8}T\d{6}Z/;

export function validateBlobPath(blobPath: string): boolean {
  return gcsAllowedBlobPrefixes().some((prefix) => blobPath.startsWith(prefix));
}

function isGcsNotFound(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { code?: number; message?: string };
  if (e.code === 404) return true;
  if (typeof e.message === "string" && e.message.includes("No such object")) return true;
  return false;
}

/**
 * Paths: .../polygon_…/<run>/<file>. When the manifest has a short run folder (e.g. …Z) but objects
 * live under an extended folder (e.g. …Z-BAGDAT-30m), parse prefix + basename to resolve the object
 * without rewriting the CSV.
 */
function parseStreetviewPolygonRunPath(blobPath: string): {
  underPolyPrefix: string;
  runSegment: string;
  basename: string;
} | null {
  const parts = blobPath.split("/").filter(Boolean);
  const polyIdx = parts.findIndex((p) => p.startsWith("polygon_"));
  if (polyIdx < 1 || polyIdx + 2 >= parts.length) return null;
  const runSegment = parts[polyIdx + 1];
  if (!CAPTURE_RUN_FOLDER_RE.test(runSegment)) return null;
  const root = parts.slice(0, polyIdx).join("/");
  const poly = parts[polyIdx];
  const basename = parts.slice(polyIdx + 2).join("/");
  const underPolyPrefix = `${root}/${poly}/`;
  return { underPolyPrefix, runSegment, basename };
}

async function listAllPrefixesPage(
  bucket: ReturnType<ReturnType<typeof getStorage>["bucket"]>,
  base: GetFilesOptions
): Promise<string[]> {
  const out: string[] = [];
  let query: GetFilesOptions = { ...base, autoPaginate: false };
  for (;;) {
    const [, nextQuery, apiResponse] = await bucket.getFiles(query);
    const pagePrefixes = (apiResponse as { prefixes?: string[] })?.prefixes ?? [];
    out.push(...pagePrefixes);
    if (!nextQuery) break;
    query = { ...nextQuery, ...base, autoPaginate: false };
  }
  return out;
}

/** On 404, find another run folder under the same polygon whose name is runSegment plus extra suffix characters. */
async function resolveCaptureRunFolderAlias(blobPath: string): Promise<string | null> {
  const parsed = parseStreetviewPolygonRunPath(blobPath);
  if (!parsed) return null;
  const { underPolyPrefix, runSegment, basename } = parsed;
  const bucket = getStorage().bucket(BUCKET_NAME);

  let prefixes: string[];
  try {
    prefixes = await listAllPrefixesPage(bucket, {
      prefix: underPolyPrefix,
      delimiter: "/",
    });
  } catch {
    return null;
  }

  const folderNames = prefixes
    .map((p) => p.slice(underPolyPrefix.length).replace(/\/$/, ""))
    .filter(Boolean);

  const extended = folderNames.filter(
    (name) => name.startsWith(runSegment) && name !== runSegment
  );
  if (extended.length === 0) return null;

  extended.sort((a, b) => b.length - a.length || a.localeCompare(b));

  for (const folder of extended) {
    const candidate = `${underPolyPrefix}${folder}/${basename}`;
    if (!validateBlobPath(candidate)) continue;
    const [exists] = await bucket.file(candidate).exists();
    if (exists) return candidate;
  }
  return null;
}

export async function downloadBlob(blobPath: string): Promise<Buffer> {
  const bucket = getStorage().bucket(BUCKET_NAME);
  try {
    const [contents] = await bucket.file(blobPath).download();
    return contents;
  } catch (err) {
    if (!isGcsNotFound(err)) throw err;
    const alt = await resolveCaptureRunFolderAlias(blobPath);
    if (!alt) throw err;
    const [contents] = await bucket.file(alt).download();
    return contents;
  }
}

export async function listBlobs(prefix: string): Promise<string[]> {
  const bucket = getStorage().bucket(BUCKET_NAME);
  const [files] = await bucket.getFiles({ prefix });
  return files.map((f) => f.name);
}

export async function downloadJSON<T>(blobPath: string): Promise<T> {
  const buf = await downloadBlob(blobPath);
  return JSON.parse(buf.toString("utf-8")) as T;
}

export async function uploadJSON(blobPath: string, data: unknown): Promise<string> {
  const bucket = getStorage().bucket(BUCKET_NAME);
  const blob = bucket.file(blobPath);
  const json = JSON.stringify(data, null, 2);
  await blob.save(json, { contentType: "application/json" });
  return `gs://${BUCKET_NAME}/${blobPath}`;
}
