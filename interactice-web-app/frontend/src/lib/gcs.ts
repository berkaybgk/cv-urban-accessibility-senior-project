import { Storage } from "@google-cloud/storage";

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

const ALLOWED_PREFIXES = [
  "streetview/",
  "segmentation-results/",
  "visualization-results/",
];

export function validateBlobPath(blobPath: string): boolean {
  return ALLOWED_PREFIXES.some((prefix) => blobPath.startsWith(prefix));
}

export async function downloadBlob(blobPath: string): Promise<Buffer> {
  const bucket = getStorage().bucket(BUCKET_NAME);
  const [contents] = await bucket.file(blobPath).download();
  return contents;
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
