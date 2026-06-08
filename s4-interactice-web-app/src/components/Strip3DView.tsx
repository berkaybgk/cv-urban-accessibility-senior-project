"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import type { StripMetadata } from "@/lib/types";

// Dynamically import SidewalkScene to prevent SSR issues with Canvas and WebGL
const SidewalkScene = dynamic(() => import("./SidewalkScene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center bg-neutral-950 text-neutral-400">
      <div className="flex flex-col items-center gap-3">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-neutral-700 border-t-transparent" />
        <span className="text-sm">Initializing 3D renderer...</span>
      </div>
    </div>
  ),
});

interface Strip3DViewProps {
  gcsPrefix: string;
}

export default function Strip3DView({ gcsPrefix }: Strip3DViewProps) {
  const [selectedSide, setSelectedSide] = useState<"left" | "right">("right");
  const [metadata, setMetadata] = useState<StripMetadata | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showGuide, setShowGuide] = useState(true);

  // Compute pxToMeter ratio dynamically based on estimated physical width
  const pxToMeter = metadata?.avg_sidewalk_width_m && metadata?.target_sidewalk_width_px
    ? metadata.avg_sidewalk_width_m / metadata.target_sidewalk_width_px
    : 0.05;

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    setMetadata(null);

    const metadataBlob = `${gcsPrefix}${selectedSide}_sidewalk_strip_METADATA.json`;
    const url = `/api/image?blob=${encodeURIComponent(metadataBlob)}`;

    fetch(url)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(
            `No 3D segment data found for the ${selectedSide} sidewalk. (HTTP ${res.status})`
          );
        }
        return res.json() as Promise<StripMetadata>;
      })
      .then((data) => {
        if (!active) return;
        setMetadata(data);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [gcsPrefix, selectedSide]);

  // Count instances of each obstacle class for the stats section
  const obstacleCounts = metadata?.boxes.reduce((acc, box) => {
    acc[box.class_name] = (acc[box.class_name] || 0) + 1;
    return acc;
  }, {} as Record<string, number>) ?? {};

  const obstacleList = Object.entries(obstacleCounts).sort((a, b) =>
    a[0].localeCompare(b[0])
  );

  const footprintImageUrl = `/api/image?blob=${encodeURIComponent(
    `${gcsPrefix}${selectedSide}_sidewalk_strip_FOOTPRINT_BOXES.png`
  )}`;

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-neutral-950">
      {/* Control bar */}
      <div className="flex flex-shrink-0 flex-col gap-3 border-b border-neutral-800 bg-neutral-900/60 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-400 font-medium">Select Side:</span>
          <div className="inline-flex rounded-md bg-neutral-800 p-0.5 border border-neutral-700">
            {(["left", "right"] as const).map((side) => (
              <button
                key={side}
                onClick={() => setSelectedSide(side)}
                className={`rounded px-3 py-1 text-xs font-semibold uppercase transition-colors ${
                  selectedSide === side
                    ? "bg-cyan-600 text-white shadow-sm"
                    : "text-neutral-400 hover:text-neutral-200"
                }`}
              >
                {side}
              </button>
            ))}
          </div>
        </div>

        {metadata && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-neutral-400">
            <div>
              Width:{" "}
              <span className="text-neutral-200 font-medium">
                {(metadata.width * pxToMeter).toFixed(1)}m
              </span>{" "}
              <span className="text-neutral-500">({metadata.width}px)</span>
            </div>
            <div className="h-3 w-[1px] bg-neutral-700" />
            <div>
              Length:{" "}
              <span className="text-neutral-200 font-medium">
                {(metadata.height * pxToMeter).toFixed(1)}m
              </span>{" "}
              <span className="text-neutral-500">({metadata.height}px)</span>
            </div>
            <div className="h-3 w-[1px] bg-neutral-700" />
            <div>
              Obstacles:{" "}
              <span className="text-cyan-400 font-semibold">
                {metadata.boxes.length} detected
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Main split display area */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* Left pane: 2D preview & Obstacle list */}
        <div className="flex flex-col border-b border-neutral-800 lg:w-80 lg:border-b-0 lg:border-r bg-neutral-900/30 overflow-hidden">
          {/* Obstacle Stats / Legend */}
          {metadata && (
            <div className="flex-shrink-0 border-b border-neutral-800/80 bg-neutral-900/50 p-4">
              <h3 className="text-xs font-semibold text-neutral-300 uppercase tracking-wider mb-2.5">
                Obstacle Count
              </h3>
              {obstacleList.length === 0 ? (
                <p className="text-xs text-neutral-500 italic">
                  No obstacles detected in this segment.
                </p>
              ) : (
                <div className="grid grid-cols-2 gap-2">
                  {obstacleList.map(([cls, count]) => {
                    let dotColor = "bg-red-400";
                    if (cls === "bollard") dotColor = "bg-[#737373]";
                    else if (cls === "trash_container") dotColor = "bg-[#3b6e4c]";
                    else if (cls === "tree") dotColor = "bg-[#226622]";
                    else if (cls === "street_sign" || cls === "traffic_sign") dotColor = "bg-[#0284c7]";
                    else if (cls === "traffic_light") dotColor = "bg-[#1e293b] border border-neutral-700";
                    else if (cls === "bench") dotColor = "bg-[#8c5630]";

                    return (
                      <div
                        key={cls}
                        className="flex items-center gap-2 rounded bg-neutral-900/50 border border-neutral-800/60 px-2 py-1.5"
                      >
                        <span className={`h-2.5 w-2.5 rounded-full ${dotColor}`} />
                        <div className="flex-1 min-w-0">
                          <div className="text-[10px] font-medium text-neutral-300 truncate capitalize">
                            {cls.replace("_", " ")}
                          </div>
                          <div className="text-[9px] text-neutral-500 font-semibold mt-0.5">
                            {count} {count === 1 ? "unit" : "units"}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Scrollable 2D Stitched footprint image */}
          <div className="flex-1 overflow-y-auto p-4 flex flex-col items-center min-h-[200px] lg:min-h-0 bg-neutral-950/20">
            <h4 className="text-[10px] font-semibold text-neutral-500 uppercase tracking-wider mb-2 self-start">
              2D Footprint Map
            </h4>
            <div className="relative border border-neutral-800 rounded bg-neutral-900/50 overflow-hidden shadow-inner p-1 max-w-[180px]">
              <img
                src={footprintImageUrl}
                alt={`${selectedSide} sidewalk footprint strip`}
                className="w-full h-auto select-none rounded"
                draggable={false}
                onError={(e) => {
                  e.currentTarget.style.display = "none";
                }}
              />
            </div>
          </div>
        </div>

        {/* Right pane: 3D Scene */}
        <div className="relative flex-1 min-h-[350px] lg:min-h-0">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-neutral-950 z-10">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-cyan-500 border-t-transparent" />
                <span className="text-sm text-neutral-400">Loading 3D segment data…</span>
              </div>
            </div>
          )}

          {error && !loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-neutral-950 p-6 text-center z-10">
              <div className="max-w-md">
                <div className="text-amber-500 text-3xl mb-3">⚠</div>
                <h4 className="text-sm font-semibold text-neutral-200 mb-1">Could Not Load 3D View</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">{error}</p>
                <div className="text-[11px] text-neutral-500 mt-4 leading-normal">
                  Make sure the strip creation has finished and includes data for this side.
                </div>
              </div>
            </div>
          )}

          {!loading && !error && metadata && (
            <div className="h-full w-full relative">
              <SidewalkScene
                boxes={metadata.boxes}
                stripHeight={metadata.height}
                stripWidth={metadata.width}
                pxToMeter={pxToMeter}
                side={selectedSide}
                avgSidewalkWidthM={metadata.avg_sidewalk_width_m}
              />

              {/* Navigation Guide Overlay */}
              {showGuide && (
                <div className="absolute bottom-4 left-4 z-10 max-w-xs rounded-lg border border-neutral-800 bg-neutral-900/90 p-3 shadow-xl backdrop-blur-md">
                  <div className="flex items-center justify-between border-b border-neutral-800 pb-1.5 mb-2">
                    <span className="text-[11px] font-semibold text-neutral-200 uppercase tracking-wide">
                      3D Navigation Controls
                    </span>
                    <button
                      onClick={() => setShowGuide(false)}
                      className="text-neutral-500 hover:text-neutral-300 text-xs px-1"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="space-y-1.5 text-[10px] text-neutral-400">
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-neutral-800 border border-neutral-700 px-1.5 py-0.5 font-mono text-[9px] text-neutral-200 shadow-sm">
                        Left Click + Drag
                      </span>
                      <span>Rotate view</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-neutral-800 border border-neutral-700 px-1.5 py-0.5 font-mono text-[9px] text-neutral-200 shadow-sm">
                        Right Click + Drag
                      </span>
                      <span>Pan along street</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-neutral-800 border border-neutral-700 px-1.5 py-0.5 font-mono text-[9px] text-neutral-200 shadow-sm">
                        Scroll / Pinch
                      </span>
                      <span>Zoom in/out</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-neutral-800 border border-neutral-700 px-1.5 py-0.5 font-mono text-[9px] text-neutral-200 shadow-sm">
                        Hover Obstacle
                      </span>
                      <span>Show details</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
