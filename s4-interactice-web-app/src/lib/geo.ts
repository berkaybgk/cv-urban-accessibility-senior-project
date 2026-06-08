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
