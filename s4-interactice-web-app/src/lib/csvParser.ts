import fs from "fs";
import path from "path";
import type { Direction, PointData, PointsHashMap } from "./types";

let cachedPoints: PointsHashMap | null = null;

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

export function getPoints(): PointsHashMap {
  if (cachedPoints) return cachedPoints;

  const csvPath = path.join(
    process.cwd(),
    "resources",
    "streetview_polygon_4v_20260221T142707Z_manifest.csv"
  );
  const raw = fs.readFileSync(csvPath, "utf-8");
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

  cachedPoints = points;
  return points;
}

export function buildCoordinateFolder(
  pointId: string,
  lat: number,
  lon: number
): string {
  return `${pointId}_${lat}_${lon}`;
}
