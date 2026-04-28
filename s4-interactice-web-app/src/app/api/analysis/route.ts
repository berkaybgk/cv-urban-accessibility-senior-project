import { NextRequest, NextResponse } from "next/server";
import { listBlobs } from "@/lib/gcs";
import { buildCoordinateFolder } from "@/lib/csvParser";
import { gcsBlobPrefix } from "@/lib/gcsPaths";
import type { AnalysisResult, AnalysisSegment, Direction } from "@/lib/types";

const ARTIFACT_MAP: Record<string, keyof AnalysisSegment["artifacts"]> = {
  "obstacle_silhouettes.png": "obstacleSilhouettes",
  "rectified_footprint.png": "rectifiedFootprint",
  "width_overlay.png": "widthOverlay",
  "width_profile.png": "widthProfile",
  "footprint_metadata.json": "footprintMetadata",
  "width_metadata.json": "widthMetadata",
};

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
  // Squared distance in degree space is sufficient for ranking candidates.
  return dLat * dLat + dLon * dLon;
}

async function resolveAnalysisPrefix(
  pointId: string,
  direction: Direction,
  lat: number | null,
  lon: number | null
): Promise<{ prefix: string; coordinateFolder: string }> {
  const visualizationRoot = gcsBlobPrefix("visualization");
  console.info(
    `[analysis] resolve start point=${pointId} dir=${direction} root=${visualizationRoot}`
  );

  // Legacy/exact first if coordinates are provided.
  if (lat != null && lon != null) {
    const exactCoord = buildCoordinateFolder(pointId, lat, lon);
    const exactPrefix = `${visualizationRoot}${exactCoord}/${direction}/`;
    console.info(`[analysis] trying exact prefix: ${exactPrefix}`);
    const exactBlobs = await listBlobs(exactPrefix);
    console.info(
      `[analysis] exact prefix result count=${exactBlobs.length} prefix=${exactPrefix}`
    );
    if (exactBlobs.length > 0) {
      console.info(`[analysis] resolved via exact prefix: ${exactPrefix}`);
      return { prefix: exactPrefix, coordinateFolder: exactCoord };
    }
  }

  // Point-id based lookup: v*/visualization-results/{point_id}_*/{direction}/...
  const pointPrefix = `${visualizationRoot}${pointId}_`;
  console.info(`[analysis] listing point candidates with prefix: ${pointPrefix}`);
  const candidateBlobs = await listBlobs(pointPrefix);
  console.info(
    `[analysis] point candidate blobs count=${candidateBlobs.length} prefix=${pointPrefix}`
  );
  const marker = `/${direction}/`;
  const coordFolders = new Set<string>();

  for (const blobName of candidateBlobs) {
    const rel = blobName.slice(visualizationRoot.length);
    const parts = rel.split("/");
    if (parts.length < 3) continue;
    const coordFolder = parts[0];
    const dir = parts[1];
    if (dir !== direction) continue;
    coordFolders.add(coordFolder);
  }

  if (coordFolders.size === 0) {
    // No matches; keep response compatible with previous behavior.
    const fallbackCoord = lat != null && lon != null
      ? buildCoordinateFolder(pointId, lat, lon)
      : `${pointId}_unknown_unknown`;
    console.warn(
      `[analysis] no candidate folders found for point=${pointId} dir=${direction}; fallback=${visualizationRoot}${fallbackCoord}/${direction}/`
    );
    return {
      prefix: `${visualizationRoot}${fallbackCoord}/${direction}/`,
      coordinateFolder: fallbackCoord,
    };
  }

  const chosenCoord = Array.from(coordFolders).sort((a, b) => {
    const sA = scoreCoordFolder(a, lat, lon);
    const sB = scoreCoordFolder(b, lat, lon);
    if (sA !== sB) return sA - sB;
    return a.localeCompare(b);
  })[0];

  return {
    prefix: `${visualizationRoot}${chosenCoord}/${direction}/`,
    coordinateFolder: chosenCoord,
  };
}

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const pointId = params.get("pointId");
  const direction = params.get("direction") as Direction | null;
  const lat = params.get("lat");
  const lon = params.get("lon");
  const originalBlob = params.get("originalBlob");

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
    const { prefix, coordinateFolder } = await resolveAnalysisPrefix(
      pointId,
      direction,
      targetLat,
      targetLon
    );
    console.info(`[analysis] final prefix=${prefix}`);
    const blobs = await listBlobs(prefix);
    console.info(
      `[analysis] fetched artifacts count=${blobs.length} for prefix=${prefix}`
    );

    const segmentMap = new Map<string, AnalysisSegment>();

    for (const blobName of blobs) {
      const relative = blobName.slice(prefix.length);
      const parts = relative.split("/");
      if (parts.length < 2) continue;

      const segmentKey = parts[0];
      const fileName = parts.slice(1).join("/");
      const artifactKey = ARTIFACT_MAP[fileName];
      if (!artifactKey) continue;

      if (!segmentMap.has(segmentKey)) {
        segmentMap.set(segmentKey, { segmentKey, artifacts: {} });
      }

      const seg = segmentMap.get(segmentKey)!;
      seg.artifacts[artifactKey] = `/api/image?blob=${encodeURIComponent(blobName)}`;
    }

    const segments = Array.from(segmentMap.values()).sort((a, b) =>
      a.segmentKey.localeCompare(b.segmentKey)
    );

    const result: AnalysisResult = {
      pointId,
      direction,
      coordinateFolder,
      originalImageUrl: originalBlob
        ? `/api/image?blob=${encodeURIComponent(originalBlob)}`
        : "",
      segments,
    };

    return NextResponse.json(result);
  } catch (err) {
    console.error(
      `Failed to fetch analysis results for point=${pointId} direction=${direction}:`,
      err
    );
    return NextResponse.json(
      { error: "Failed to fetch analysis results" },
      { status: 500 }
    );
  }
}
