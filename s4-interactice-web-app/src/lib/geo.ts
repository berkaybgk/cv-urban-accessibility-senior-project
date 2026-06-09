export type LngLat = [number, number];

/** Color buckets for accessibility scores (higher score = greener). */
export const SCORE_COLORS = {
  green: "#22c55e",
  yellow: "#eab308",
  red: "#ef4444",
} as const;

/** Score thresholds on a 0–1 scale: red < LOW <= yellow < HIGH <= green. */
export const SCORE_THRESHOLDS = { low: 0.33, high: 0.66 } as const;

export interface ScoreBucket {
  label: string;
  color: string;
  range: string;
}

/** Which per-segment score drives the map colors. */
export type ScoreMetric = "walkability_score" | "wheelchair_score";

export const METRIC_LABELS: Record<ScoreMetric, string> = {
  walkability_score: "Walkability",
  wheelchair_score: "Wheelchair",
};

/** Legend entries, ordered best → worst. Shared by the panel and the PNG legend. */
export const SCORE_LEGEND: ScoreBucket[] = [
  { label: "Accessible", color: SCORE_COLORS.green, range: `≥ ${SCORE_THRESHOLDS.high}` },
  { label: "Marginal", color: SCORE_COLORS.yellow, range: `${SCORE_THRESHOLDS.low}–${SCORE_THRESHOLDS.high}` },
  { label: "Not accessible", color: SCORE_COLORS.red, range: `< ${SCORE_THRESHOLDS.low}` },
];

export function colorForScore(score: number): string {
  if (score >= SCORE_THRESHOLDS.high) return SCORE_COLORS.green;
  if (score >= SCORE_THRESHOLDS.low) return SCORE_COLORS.yellow;
  return SCORE_COLORS.red;
}

/** Ray-casting point-in-polygon test. `ring` is an array of [lng, lat] vertices (not closed). */
export function pointInPolygon(lng: number, lat: number, ring: LngLat[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const intersects =
      yi > lat !== yj > lat &&
      lng < ((xj - xi) * (lat - yi)) / (yj - yi + Number.EPSILON) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

/**
 * True if a LineString lies (at least partly) inside the polygon ring.
 * A segment counts as inside if any of its vertices is inside — good enough
 * for poster figures where segments are short relative to the area.
 */
export function segmentInPolygon(coords: LngLat[], ring: LngLat[]): boolean {
  if (ring.length < 3) return false;
  return coords.some(([lng, lat]) => pointInPolygon(lng, lat, ring));
}

/** Bounding box [minLng, minLat, maxLng, maxLat] of a ring of [lng, lat] vertices. */
export function bboxOf(ring: LngLat[]): [number, number, number, number] {
  let minLng = Infinity;
  let minLat = Infinity;
  let maxLng = -Infinity;
  let maxLat = -Infinity;
  for (const [lng, lat] of ring) {
    if (lng < minLng) minLng = lng;
    if (lat < minLat) minLat = lat;
    if (lng > maxLng) maxLng = lng;
    if (lat > maxLat) maxLat = lat;
  }
  return [minLng, minLat, maxLng, maxLat];
}

export function haversineDistance(coord1: LngLat, coord2: LngLat): number {
  const R = 6371e3; // Earth radius in meters
  const lat1 = (coord1[1] * Math.PI) / 180;
  const lat2 = (coord2[1] * Math.PI) / 180;
  const deltaLat = ((coord2[1] - coord1[1]) * Math.PI) / 180;
  const deltaLng = ((coord2[0] - coord1[0]) * Math.PI) / 180;

  const a =
    Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(deltaLng / 2) * Math.sin(deltaLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c; // in meters
}

export function calculateLineStringLength(coords: LngLat[]): number {
  let total = 0;
  for (let i = 0; i < coords.length - 1; i++) {
    total += haversineDistance(coords[i], coords[i + 1]);
  }
  return total;
}

export function getLineStringMidpoint(coordinates: LngLat[]): LngLat {
  if (coordinates.length === 0) return [0, 0];
  if (coordinates.length === 1) return coordinates[0];

  let totalLength = 0;
  const lengths: number[] = [];
  for (let i = 0; i < coordinates.length - 1; i++) {
    const dist = haversineDistance(coordinates[i], coordinates[i + 1]);
    lengths.push(dist);
    totalLength += dist;
  }

  const targetLength = totalLength / 2;
  let currentLength = 0;

  for (let i = 0; i < coordinates.length - 1; i++) {
    const dist = lengths[i];
    if (currentLength + dist >= targetLength) {
      const ratio = (targetLength - currentLength) / (dist || 1);
      const [lng1, lat1] = coordinates[i];
      const [lng2, lat2] = coordinates[i + 1];
      return [
        lng1 + (lng2 - lng1) * ratio,
        lat1 + (lat2 - lat1) * ratio
      ];
    }
    currentLength += dist;
  }

  return coordinates[coordinates.length - 1];
}
