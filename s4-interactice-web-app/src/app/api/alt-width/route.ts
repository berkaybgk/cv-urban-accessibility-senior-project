import { NextRequest, NextResponse } from "next/server";
import { listBlobs } from "@/lib/gcs";
import { buildCoordinateFolder } from "@/lib/csvParser";
import { gcsBlobPrefix } from "@/lib/gcsPaths";
import type { AlternativeWidthResult, Direction } from "@/lib/types";

function parseCoordFolderLatLon(coordFolder: string): { lat: number; lon: number } | null {
  const m = coordFolder.match(/^\d+_([-\d.]+)_([-\d.]+)$/);
  if (!m) return null;
  const lat = Number.parseFloat(m[1]);
  const lon = Number.parseFloat(m[2]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return { lat, lon };
}

function scoreCoordFolder(
  coordFolder: string,
  targetLat: number | null,
  targetLon: number | null
): number {
  if (targetLat == null || targetLon == null) return 0;
  const parsed = parseCoordFolderLatLon(coordFolder);
  if (!parsed) return Number.POSITIVE_INFINITY;
  const dLat = parsed.lat - targetLat;
  const dLon = parsed.lon - targetLon;
  return dLat * dLat + dLon * dLon;
}

async function resolvePrefix(
  pointId: string,
  direction: Direction,
  lat: number | null,
  lon: number | null
): Promise<{ prefix: string; coordinateFolder: string }> {
  const root = gcsBlobPrefix("alternativeWidth");
  if (lat != null && lon != null) {
    const exactCoord = buildCoordinateFolder(pointId, lat, lon);
    const exactPrefix = `${root}${exactCoord}/${direction}/`;
    const exact = await listBlobs(exactPrefix);
    if (exact.length > 0) {
      return { prefix: exactPrefix, coordinateFolder: exactCoord };
    }
  }

  const pointPrefix = `${root}${pointId}_`;
  const blobs = await listBlobs(pointPrefix);
  const coordFolders = new Set<string>();
  for (const blobName of blobs) {
    const rel = blobName.slice(root.length);
    const parts = rel.split("/");
    if (parts.length < 3) continue;
    if (parts[1] !== direction) continue;
    coordFolders.add(parts[0]);
  }

  if (coordFolders.size === 0) {
    const fallback = lat != null && lon != null
      ? buildCoordinateFolder(pointId, lat, lon)
      : `${pointId}_unknown_unknown`;
    return {
      prefix: `${root}${fallback}/${direction}/`,
      coordinateFolder: fallback,
    };
  }

  const chosen = Array.from(coordFolders).sort((a, b) => {
    const sA = scoreCoordFolder(a, lat, lon);
    const sB = scoreCoordFolder(b, lat, lon);
    if (sA !== sB) return sA - sB;
    return a.localeCompare(b);
  })[0];

  return {
    prefix: `${root}${chosen}/${direction}/`,
    coordinateFolder: chosen,
  };
}

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const pointId = params.get("pointId");
  const direction = params.get("direction") as Direction | null;
  const lat = params.get("lat");
  const lon = params.get("lon");

  if (!pointId || !direction) {
    return NextResponse.json(
      { error: "Missing required query parameters: pointId, direction" },
      { status: 400 }
    );
  }

  const parsedLat = lat != null ? Number.parseFloat(lat) : Number.NaN;
  const parsedLon = lon != null ? Number.parseFloat(lon) : Number.NaN;
  const targetLat = Number.isFinite(parsedLat) ? parsedLat : null;
  const targetLon = Number.isFinite(parsedLon) ? parsedLon : null;

  try {
    const { prefix, coordinateFolder } = await resolvePrefix(
      pointId,
      direction,
      targetLat,
      targetLon
    );
    const blobs = await listBlobs(prefix);
    const getUrl = (name: string) => {
      const blob = blobs.find((b) => b.endsWith(`/${name}`));
      return blob ? `/api/image?blob=${encodeURIComponent(blob)}` : undefined;
    };

    const result: AlternativeWidthResult = {
      pointId,
      direction,
      coordinateFolder,
      metadataUrl: getUrl("alt_width_metadata.json"),
      overlayUrl: getUrl("alt_width_overlay.png"),
      scene3dUrl: getUrl("alt_width_scene3d.png"),
      width3dUrl: getUrl("alt_width_width3d.png"),
    };
    return NextResponse.json(result);
  } catch (err) {
    console.error(
      `Failed to fetch alternative width for point=${pointId} direction=${direction}:`,
      err
    );
    return NextResponse.json(
      { error: "Failed to fetch alternative width" },
      { status: 500 }
    );
  }
}
