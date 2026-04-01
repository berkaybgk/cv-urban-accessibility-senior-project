import fs from "fs";
import path from "path";
import { downloadBlob, validateBlobPath } from "@/lib/gcs";
import type { Direction, PointsHashMap } from "./types";

let pointsCache: Promise<PointsHashMap> | null = null;

/** Relative to project root unless absolute. Used when POINTS_MANIFEST_BLOB is unset. */
const DEFAULT_POINTS_MANIFEST = "resources/streetview_polygon_4v_20260329T023321Z-BAGDAT-30m_manifest.csv";

const BUCKET_NAME = process.env.GCS_BUCKET_NAME || "cv-urban-accessibility-bucket";

function resolvePointsManifestPath(): string {
  const fromEnv = process.env.POINTS_MANIFEST_CSV?.trim();
  const rel = fromEnv || DEFAULT_POINTS_MANIFEST;
  return path.isAbsolute(rel) ? rel : path.join(process.cwd(), rel);
}

/**
 * Object key under GCS_BUCKET_NAME, or full gs://bucket/key (bucket must match env).
 */
function resolveManifestBlobKey(): string | null {
  const raw = process.env.POINTS_MANIFEST_BLOB?.trim();
  if (!raw) return null;
  const m = raw.match(/^gs:\/\/([^/]+)\/(.+)$/);
  if (m) {
    const [, bucket, key] = m;
    if (bucket !== BUCKET_NAME) {
      throw new Error(
        `POINTS_MANIFEST_BLOB uses bucket "${bucket}" but GCS_BUCKET_NAME is "${BUCKET_NAME}"`
      );
    }
    return key.replace(/^\/+/, "");
  }
  return raw.replace(/^\/+/, "");
}

function parseCSVLine(line: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;

  for (const ch of line) {
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      fields.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }
  fields.push(current.trim());
  return fields;
}

function stripGcsPrefix(uri: string): string {
  const match = uri.match(/^gs:\/\/[^/]+\/(.+)$/);
  return match ? match[1] : uri;
}

function parseManifestCsv(raw: string): PointsHashMap {
  const lines = raw.split("\n").filter((l) => l.trim().length > 0);

  const header = parseCSVLine(lines[0]);
  const colIdx: Record<string, number> = {};
  header.forEach((col, i) => {
    colIdx[col] = i;
  });

  const points: PointsHashMap = {};

  for (let i = 1; i < lines.length; i++) {
    const fields = parseCSVLine(lines[i]);
    if (fields.length < header.length) continue;

    const pointId = fields[colIdx["point_id"]];
    const direction = fields[colIdx["direction"]] as Direction;
    const gcsUri = fields[colIdx["gcs_uri"]];
    const latitude = parseFloat(fields[colIdx["latitude"]]);
    const longitude = parseFloat(fields[colIdx["longitude"]]);
    const streetBearing = parseFloat(fields[colIdx["street_bearing"]]);
    const heading = parseFloat(fields[colIdx["heading"]]);
    const panoId = fields[colIdx["pano_id"]];

    if (!points[pointId]) {
      points[pointId] = {
        pointId,
        latitude,
        longitude,
        streetBearing,
        panoId,
        directions: {},
      };
    }

    points[pointId].directions[direction] = {
      gcsUri: stripGcsPrefix(gcsUri),
      heading,
    };
  }

  return points;
}

async function loadPoints(): Promise<PointsHashMap> {
  const blobKey = resolveManifestBlobKey();
  let raw: string;

  if (blobKey) {
    if (!validateBlobPath(blobKey)) {
      throw new Error(
        `Manifest object key is not under an allowed prefix (see gcsPaths / GCS_EXTRA_ALLOWED_PREFIXES): ${blobKey}`
      );
    }
    const buf = await downloadBlob(blobKey);
    raw = buf.toString("utf-8");
  } else {
    const csvPath = resolvePointsManifestPath();
    raw = fs.readFileSync(csvPath, "utf-8");
  }

  return parseManifestCsv(raw);
}

/**
 * Loads the streetview manifest once per process (cached).
 * Prefers `POINTS_MANIFEST_BLOB` (GCS object key or gs:// URI); otherwise reads `POINTS_MANIFEST_CSV` on disk.
 */
export function getPoints(): Promise<PointsHashMap> {
  if (!pointsCache) {
    pointsCache = loadPoints();
  }
  return pointsCache;
}

export function buildCoordinateFolder(
  pointId: string,
  lat: number,
  lon: number
): string {
  return `${pointId}_${lat}_${lon}`;
}
