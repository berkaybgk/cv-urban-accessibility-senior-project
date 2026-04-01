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

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const pointId = params.get("pointId");
  const direction = params.get("direction") as Direction | null;
  const lat = params.get("lat");
  const lon = params.get("lon");
  const originalBlob = params.get("originalBlob");

  if (!pointId || !direction || !lat || !lon) {
    return NextResponse.json(
      { error: "Missing required query parameters: pointId, direction, lat, lon" },
      { status: 400 }
    );
  }

  const coordFolder = buildCoordinateFolder(pointId, parseFloat(lat), parseFloat(lon));
  const prefix = `${gcsBlobPrefix("visualization")}${coordFolder}/${direction}/`;

  try {
    const blobs = await listBlobs(prefix);

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
      coordinateFolder: coordFolder,
      originalImageUrl: originalBlob
        ? `/api/image?blob=${encodeURIComponent(originalBlob)}`
        : "",
      segments,
    };

    return NextResponse.json(result);
  } catch (err) {
    console.error(`Failed to list analysis results for ${prefix}:`, err);
    return NextResponse.json(
      { error: "Failed to fetch analysis results" },
      { status: 500 }
    );
  }
}
