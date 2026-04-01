/**
 * Single place to configure GCS object key roots. Override per deployment with env vars.
 *
 * Values are folder paths without a leading slash. Trailing slashes are normalized away.
 */
function envPath(key: string, fallback: string): string {
  const v = process.env[key]?.trim();
  if (!v) return fallback;
  return v.replace(/^\/+/, "").replace(/\/+$/, "");
}

export const gcsPaths = {
  /**
   * Original / street-view imagery. Default `streetview` allows any subpath
   * (e.g. polygon_4v/20260329T023321Z-BAGDAT-30m/… under streetview/). Set a deeper
   * prefix to lock to one capture run (must match manifest gs:// object keys).
   */
  original: envPath("GCS_PREFIX_ORIGINAL", "streetview"),
  segmentation: envPath("GCS_PREFIX_SEGMENTATION", "v2/segmentation-results"),
  visualization: envPath("GCS_PREFIX_VISUALIZATION", "v2/visualization-results"),
  reports: envPath("GCS_PREFIX_REPORTS", "reports"),
} as const;

export type GcsPathKey = keyof typeof gcsPaths;

/** Object key prefix with trailing slash (for prefix listing and startsWith checks). */
export function gcsBlobPrefix(key: GcsPathKey): string {
  return `${gcsPaths[key]}/`;
}

/** Object key segment without trailing slash (for `${gcsPathSegment("visualization")}/foo`). */
export function gcsPathSegment(key: GcsPathKey): string {
  return gcsPaths[key];
}

/** Prefixes the image API is allowed to serve (keep in sync with what you store in the bucket). */
export function gcsAllowedBlobPrefixes(): string[] {
  const extra = process.env.GCS_EXTRA_ALLOWED_PREFIXES?.split(",") ?? [];
  const normalized = extra
    .map((s) => s.trim().replace(/^\/+/, ""))
    .filter(Boolean)
    .map((p) => (p.endsWith("/") ? p : `${p}/`));
  const base: GcsPathKey[] = ["original", "segmentation", "visualization", "reports"];
  return [...base.map((k) => gcsBlobPrefix(k)), ...normalized];
}
