"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  StripEdgeHandle,
  StripEdgeLines,
  StripEdgesResult,
  StripTilePreview,
} from "@/lib/types";

interface StripBuilderProps {
  open: boolean;
  pointIds: string[];
  onClose: () => void;
}

type DragTarget = { side: "left" | "right"; end: "top" | "bottom" };

interface CreateResult {
  stripId: string;
  gcsPrefix: string;
}

const RESULT_PREVIEWS = [
  "left_sidewalk_strip_MERGED.png",
  "right_sidewalk_strip_MERGED.png",
  "left_sidewalk_strip_FOOTPRINT_BOXES.png",
  "right_sidewalk_strip_FOOTPRINT_BOXES.png",
];

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export default function StripBuilder({ open, pointIds, onClose }: StripBuilderProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [edges, setEdges] = useState<StripEdgesResult | null>(null);
  const [index, setIndex] = useState(0);
  const [overrides, setOverrides] = useState<Record<string, StripEdgeLines>>({});

  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [result, setResult] = useState<CreateResult | null>(null);

  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<DragTarget | null>(null);

  // Fetch per-tile editable lines when opened.
  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    setEdges(null);
    setOverrides({});
    setIndex(0);
    setResult(null);
    setCreateError(null);

    fetch("/api/strip/edges", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pointIds }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || body.error || `Request failed (${res.status})`);
        }
        return res.json() as Promise<StripEdgesResult>;
      })
      .then((data) => setEdges(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [open, pointIds]);

  const tiles = edges?.tiles ?? [];
  const tile: StripTilePreview | undefined = tiles[index];
  const currentLines: StripEdgeLines | undefined = tile
    ? overrides[tile.tileId] ?? tile.lines
    : undefined;
  const isOverridden = tile ? Boolean(overrides[tile.tileId]) : false;

  const setHandle = useCallback(
    (tileId: string, base: StripEdgeLines, side: "left" | "right", end: "top" | "bottom", x: number, y: number) => {
      setOverrides((prev) => {
        const existing = prev[tileId] ?? base;
        const handle: StripEdgeHandle = { ...existing[side] };
        if (end === "top") {
          handle.xTop = x;
          handle.yTop = y;
        } else {
          handle.xBottom = x;
          handle.yBottom = y;
        }
        return { ...prev, [tileId]: { ...existing, [side]: handle } };
      });
    },
    []
  );

  const eventToFrame = useCallback((evt: React.PointerEvent | PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return null;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    const local = pt.matrixTransform(ctm.inverse());
    return { x: local.x, y: local.y };
  }, []);

  const onPointerMove = useCallback(
    (evt: React.PointerEvent<SVGSVGElement>) => {
      const drag = dragRef.current;
      if (!drag || !tile || !tile.frame || !currentLines) return;
      const p = eventToFrame(evt);
      if (!p) return;
      setHandle(
        tile.tileId,
        tile.lines!,
        drag.side,
        drag.end,
        clamp(p.x, 0, tile.frame.w),
        clamp(p.y, 0, tile.frame.h)
      );
    },
    [tile, currentLines, eventToFrame, setHandle]
  );

  const endDrag = useCallback(() => {
    dragRef.current = null;
  }, []);

  const resetTile = useCallback(() => {
    if (!tile) return;
    setOverrides((prev) => {
      const next = { ...prev };
      delete next[tile.tileId];
      return next;
    });
  }, [tile]);

  const handleCreate = useCallback(async () => {
    setCreating(true);
    setCreateError(null);
    setResult(null);
    try {
      const res = await fetch("/api/strip/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pointIds, overrides }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || body.error || `Request failed (${res.status})`);
      }
      const stripId = res.headers.get("X-Strip-Id") || "";
      const gcsPrefix = res.headers.get("X-Strip-Gcs-Prefix") || "";
      const blob = await res.blob();

      // Trigger browser download of the zip.
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `strip_${stripId || "output"}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      setResult({ stripId, gcsPrefix });
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  }, [pointIds, overrides]);

  const overrideCount = Object.keys(overrides).length;

  const previewSrc = useMemo(() => {
    if (!tile?.previewPng) return null;
    return `data:image/png;base64,${tile.previewPng}`;
  }, [tile]);

  if (!open) return null;

  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="flex h-[88vh] w-[min(1100px,94vw)] flex-col overflow-hidden rounded-xl border border-neutral-700 bg-neutral-900 text-neutral-100 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-neutral-700 px-5 py-3">
          <div>
            <h2 className="text-sm font-semibold">Build sidewalk strip</h2>
            <p className="text-xs text-neutral-400">
              {pointIds.length} points
              {edges ? ` · order: ${edges.orderedPoints.join(" → ")}` : ""}
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md px-2 py-1 text-neutral-400 hover:bg-neutral-800 hover:text-white"
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div className="flex min-h-0 flex-1">
          {loading && (
            <div className="flex flex-1 items-center justify-center text-sm text-neutral-400">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-neutral-600 border-t-transparent" />
                Computing rectification lines for each image…
              </div>
            </div>
          )}

          {error && !loading && (
            <div className="flex flex-1 items-center justify-center p-6 text-center text-sm text-red-400">
              {error}
            </div>
          )}

          {!loading && !error && tile && (
            <>
              {/* Editor */}
              <div className="flex min-w-0 flex-1 flex-col items-center justify-center gap-3 border-r border-neutral-800 p-4">
                <div className="flex w-full items-center justify-between text-xs text-neutral-400">
                  <button
                    onClick={() => setIndex((i) => Math.max(0, i - 1))}
                    disabled={index === 0}
                    className="rounded-md border border-neutral-700 px-3 py-1 enabled:hover:bg-neutral-800 disabled:opacity-40"
                  >
                    ‹ Prev
                  </button>
                  <span>
                    Tile {index + 1} / {tiles.length} —{" "}
                    <span className="text-neutral-200">
                      {tile.side} · pt {tile.pointId} · {tile.direction}
                    </span>
                  </span>
                  <button
                    onClick={() => setIndex((i) => Math.min(tiles.length - 1, i + 1))}
                    disabled={index === tiles.length - 1}
                    className="rounded-md border border-neutral-700 px-3 py-1 enabled:hover:bg-neutral-800 disabled:opacity-40"
                  >
                    Next ›
                  </button>
                </div>

                {tile.error ? (
                  <div className="flex flex-1 items-center justify-center px-6 text-center text-sm text-amber-400">
                    This tile is unavailable: {tile.error}
                  </div>
                ) : (
                  <div className="relative flex max-h-full items-center justify-center">
                    {previewSrc && tile.frame && (
                      <img
                        src={previewSrc}
                        alt={tile.tileId}
                        className="max-h-[58vh] w-auto select-none rounded-md"
                        draggable={false}
                      />
                    )}
                    {tile.frame && currentLines && (
                      <svg
                        ref={svgRef}
                        viewBox={`0 0 ${tile.frame.w} ${tile.frame.h}`}
                        preserveAspectRatio="none"
                        className="absolute inset-0 h-full w-full touch-none"
                        onPointerMove={onPointerMove}
                        onPointerUp={endDrag}
                        onPointerLeave={endDrag}
                      >
                        {(["left", "right"] as const).map((side) => {
                          const h = currentLines[side];
                          const color = side === "left" ? "#22c55e" : "#ef4444";
                          return (
                            <g key={side}>
                              <line
                                x1={h.xTop}
                                y1={h.yTop}
                                x2={h.xBottom}
                                y2={h.yBottom}
                                stroke={color}
                                strokeWidth={3}
                                vectorEffect="non-scaling-stroke"
                              />
                              {(["top", "bottom"] as const).map((end) => {
                                const cx = end === "top" ? h.xTop : h.xBottom;
                                const cy = end === "top" ? h.yTop : h.yBottom;
                                return (
                                  <circle
                                    key={end}
                                    cx={cx}
                                    cy={cy}
                                    r={8}
                                    fill={color}
                                    stroke="#fff"
                                    strokeWidth={2}
                                    vectorEffect="non-scaling-stroke"
                                    style={{ cursor: "grab" }}
                                    onPointerDown={(e) => {
                                      e.currentTarget.setPointerCapture(e.pointerId);
                                      dragRef.current = { side, end };
                                    }}
                                  />
                                );
                              })}
                            </g>
                          );
                        })}
                      </svg>
                    )}
                  </div>
                )}

                <div className="flex items-center gap-3 text-xs">
                  <span className="text-neutral-500">
                    <span className="text-green-400">●</span> left edge ·{" "}
                    <span className="text-red-400">●</span> right edge — drag the handles
                  </span>
                  <button
                    onClick={resetTile}
                    disabled={!isOverridden}
                    className="rounded-md border border-neutral-700 px-2 py-1 enabled:hover:bg-neutral-800 disabled:opacity-40"
                  >
                    Reset to auto
                  </button>
                </div>
              </div>

              {/* Sidebar: create + result */}
              <div className="flex w-72 flex-col gap-3 p-4">
                <div className="text-xs text-neutral-400">
                  {overrideCount > 0
                    ? `${overrideCount} tile${overrideCount > 1 ? "s" : ""} manually adjusted.`
                    : "All tiles use auto-fitted lines."}
                </div>
                <button
                  onClick={handleCreate}
                  disabled={creating}
                  className="rounded-md bg-cyan-600 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-500 disabled:opacity-50"
                >
                  {creating ? "Building strip…" : "Create strip"}
                </button>
                {createError && (
                  <div className="rounded-md border border-red-800 bg-red-950/50 p-2 text-xs text-red-300">
                    {createError}
                  </div>
                )}
                {result && (
                  <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
                    <div className="rounded-md border border-emerald-800 bg-emerald-950/40 p-2 text-xs text-emerald-300">
                      Strip created &amp; downloaded.
                      <div className="mt-1 break-all text-emerald-400/80">
                        GCS: {result.gcsPrefix || "(not uploaded)"}
                      </div>
                    </div>
                    {result.gcsPrefix &&
                      RESULT_PREVIEWS.map((name) => (
                        <div key={name} className="space-y-1">
                          <div className="text-[11px] text-neutral-500">{name}</div>
                          <img
                            src={`/api/image?blob=${encodeURIComponent(result.gcsPrefix + name)}`}
                            alt={name}
                            className="w-full rounded border border-neutral-800"
                          />
                        </div>
                      ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
