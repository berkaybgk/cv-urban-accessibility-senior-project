"use client";

import { useRef } from "react";
import { SCORE_LEGEND, METRIC_LABELS, type ScoreMetric } from "@/lib/geo";

interface AccessibilityPanelProps {
  segmentCount: number | null;
  loadError: string | null;
  areaPointCount: number;
  metric: ScoreMetric;
  onMetricChange: (metric: ScoreMetric) => void;
  onLoadFiles: (files: FileList) => void;
  onClearSegments: () => void;
  onClearArea: () => void;
  onExport: () => void;
  exporting: boolean;
  hidePoints: boolean;
  onToggleHidePoints: () => void;
  isDark?: boolean;
}

const METRICS: ScoreMetric[] = ["walkability_score", "wheelchair_score"];

export default function AccessibilityPanel({
  segmentCount,
  loadError,
  areaPointCount,
  metric,
  onMetricChange,
  onLoadFiles,
  onClearSegments,
  onClearArea,
  onExport,
  exporting,
  hidePoints,
  onToggleHidePoints,
  isDark = true,
}: AccessibilityPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canExport = segmentCount !== null && areaPointCount === 4 && !exporting;

  return (
    <div className={`absolute right-4 top-4 z-20 w-72 rounded-lg border p-3 backdrop-blur-sm transition-colors duration-200 ${
      isDark
        ? "border-neutral-700/60 bg-neutral-900/90 text-neutral-100"
        : "border-neutral-200 bg-white/95 text-neutral-800 shadow-lg"
    }`}>
      <div className="mb-2 text-sm font-semibold">Accessibility map</div>

      {/* Metric toggle */}
      <div className={`mb-2 flex gap-1 rounded-md p-0.5 transition-colors duration-200 ${
        isDark ? "bg-neutral-800" : "bg-neutral-100"
      }`}>
        {METRICS.map((m) => (
          <button
            key={m}
            onClick={() => onMetricChange(m)}
            className={`flex-1 rounded px-2 py-1 text-xs font-medium transition-all ${
              metric === m
                ? "bg-emerald-600 text-white shadow-sm"
                : isDark
                  ? "text-neutral-300 hover:bg-neutral-700"
                  : "text-neutral-600 hover:bg-neutral-200"
            }`}
          >
            {METRIC_LABELS[m]}
          </button>
        ))}
      </div>

      {/* 1. Load scored GeoJSON (one or many) */}
      <button
        onClick={() => fileInputRef.current?.click()}
        className={`mb-1 w-full rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
          isDark
            ? "bg-neutral-700 text-white hover:bg-neutral-600"
            : "bg-neutral-100 text-neutral-700 hover:bg-neutral-200 border border-neutral-200"
        }`}
      >
        Load scored GeoJSON (one or many)
      </button>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".geojson,.json,application/geo+json,application/json"
        className="hidden"
        onChange={(e) => {
          const files = e.target.files;
          if (files && files.length > 0) onLoadFiles(files);
          e.target.value = "";
        }}
      />
      <p className={`mb-2 text-[11px] transition-colors ${
        isDark ? "text-neutral-400" : "text-neutral-500"
      }`}>
        {segmentCount === null
          ? "Pick a metric, then load LineString features. Multiple files merge together."
          : `${segmentCount} segments loaded. Loading more adds to these.`}
      </p>
      {loadError && (
        <p className="mb-2 text-[11px] text-red-400">{loadError}</p>
      )}
      {segmentCount !== null && (
        <button
          onClick={onClearSegments}
          className={`mb-2 w-full rounded-md border px-2 py-1 text-[11px] transition-colors ${
            isDark
              ? "border-neutral-700 hover:bg-neutral-800 text-neutral-300"
              : "border-neutral-300 hover:bg-neutral-100 text-neutral-600"
          }`}
        >
          Clear loaded segments
        </button>
      )}

      {/* 2. Pick area corners */}
      <p className={`mb-2 text-xs transition-colors ${
        isDark ? "text-neutral-300" : "text-neutral-600"
      }`}>
        Pick 4 corners on the map:{" "}
        <span className="font-semibold">{areaPointCount}/4</span>
      </p>

      {/* Hide point markers (cleaner PNG export) */}
      <button
        onClick={onToggleHidePoints}
        className={`mb-3 w-full rounded-md border px-2 py-1 text-[11px] transition-colors ${
          isDark
            ? "border-neutral-700 hover:bg-neutral-800 text-neutral-300"
            : "border-neutral-300 hover:bg-neutral-100 text-neutral-600"
        }`}
      >
        {hidePoints ? "Show point markers" : "Hide point markers"}
      </button>

      {/* Legend */}
      <div className="mb-3 space-y-1">
        {SCORE_LEGEND.map((b) => (
          <div key={b.label} className="flex items-center gap-2 text-[11px]">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ backgroundColor: b.color }}
            />
            <span className={isDark ? "text-neutral-200" : "text-neutral-700"}>{b.label}</span>
            <span className={isDark ? "text-neutral-500" : "text-neutral-400"}>{b.range}</span>
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
            className={`rounded-md border px-2 py-1.5 text-xs transition-colors ${
              isDark
                ? "border-neutral-700 hover:bg-neutral-800 text-neutral-300"
                : "border-neutral-300 hover:bg-neutral-100 text-neutral-600"
            }`}
          >
            Clear area
          </button>
        )}
      </div>
    </div>
  );
}
