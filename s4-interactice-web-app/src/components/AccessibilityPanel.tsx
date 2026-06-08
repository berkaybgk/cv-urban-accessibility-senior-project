"use client";

import { useRef } from "react";
import { SCORE_LEGEND } from "@/lib/geo";

interface AccessibilityPanelProps {
  segmentCount: number | null;
  loadError: string | null;
  areaPointCount: number;
  onLoadFile: (file: File) => void;
  onClearArea: () => void;
  onExport: () => void;
  exporting: boolean;
}

export default function AccessibilityPanel({
  segmentCount,
  loadError,
  areaPointCount,
  onLoadFile,
  onClearArea,
  onExport,
  exporting,
}: AccessibilityPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canExport = segmentCount !== null && areaPointCount === 4 && !exporting;

  return (
    <div className="absolute right-4 top-4 z-20 w-72 rounded-lg border border-neutral-700/60 bg-neutral-900/90 p-3 text-neutral-100 backdrop-blur-sm">
      <div className="mb-2 text-sm font-semibold">Accessibility map</div>

      {/* 1. Load scored GeoJSON */}
      <button
        onClick={() => fileInputRef.current?.click()}
        className="mb-1 w-full rounded-md bg-neutral-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-neutral-600"
      >
        Load scored GeoJSON
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept=".geojson,.json,application/geo+json,application/json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onLoadFile(file);
          e.target.value = "";
        }}
      />
      <p className="mb-2 text-[11px] text-neutral-400">
        {segmentCount === null
          ? "Expects LineString features with a score (0–1)."
          : `${segmentCount} segments loaded.`}
      </p>
      {loadError && (
        <p className="mb-2 text-[11px] text-red-400">{loadError}</p>
      )}

      {/* 2. Pick area corners */}
      <p className="mb-2 text-xs text-neutral-300">
        Pick 4 corners on the map:{" "}
        <span className="font-semibold">{areaPointCount}/4</span>
      </p>

      {/* Legend */}
      <div className="mb-3 space-y-1">
        {SCORE_LEGEND.map((b) => (
          <div key={b.label} className="flex items-center gap-2 text-[11px]">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ backgroundColor: b.color }}
            />
            <span className="text-neutral-200">{b.label}</span>
            <span className="text-neutral-500">{b.range}</span>
          </div>
        ))}
      </div>

      {/* 3. Actions */}
      <div className="flex gap-2">
        <button
          onClick={onExport}
          disabled={!canExport}
          className="flex-1 rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-40"
        >
          {exporting ? "Exporting…" : "Export PNG"}
        </button>
        {areaPointCount > 0 && (
          <button
            onClick={onClearArea}
            className="rounded-md border border-neutral-700 px-2 py-1.5 text-xs hover:bg-neutral-800"
          >
            Clear area
          </button>
        )}
      </div>
    </div>
  );
}
