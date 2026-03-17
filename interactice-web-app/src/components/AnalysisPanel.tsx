"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AnalysisResult,
  AnalysisSegment,
  Direction,
  FootprintMetadata,
  PointData,
  WidthMetadata,
} from "@/lib/types";
import ImageViewer from "./ImageViewer";

const ALL_DIRECTIONS: Direction[] = ["forward", "right", "backward", "left"];

const DIRECTION_LABELS: Record<Direction, string> = {
  forward: "Forward",
  right: "Right",
  backward: "Backward",
  left: "Left",
};

const DIRECTION_ARROWS: Record<Direction, string> = {
  forward: "\u2191",
  right: "\u2192",
  backward: "\u2193",
  left: "\u2190",
};

interface AnalysisPanelProps {
  open: boolean;
  onClose: () => void;
  point: PointData | null;
  direction: Direction | null;
  analysis: AnalysisResult | null;
  loading: boolean;
  onChangeDirection: (direction: Direction) => void;
}

const MIN_WIDTH = 380;
const DEFAULT_WIDTH = 480;
const MAX_WIDTH_RATIO = 0.85;

function SegmentView({
  segment,
  originalImageUrl,
}: {
  segment: AnalysisSegment;
  originalImageUrl: string;
}) {
  const [activeTab, setActiveTab] = useState("original");
  const [widthMeta, setWidthMeta] = useState<WidthMetadata | null>(null);
  const [footprintMeta, setFootprintMeta] = useState<FootprintMetadata | null>(
    null
  );

  const { artifacts } = segment;

  useEffect(() => {
    if (artifacts.widthMetadata) {
      fetch(artifacts.widthMetadata)
        .then((r) => r.json())
        .then(setWidthMeta)
        .catch(() => setWidthMeta(null));
    }
    if (artifacts.footprintMetadata) {
      fetch(artifacts.footprintMetadata)
        .then((r) => r.json())
        .then(setFootprintMeta)
        .catch(() => setFootprintMeta(null));
    }
  }, [artifacts.widthMetadata, artifacts.footprintMetadata]);

  const tabs = [
    { id: "original", label: "Original", available: !!originalImageUrl },
    {
      id: "silhouettes",
      label: "Silhouettes",
      available: !!artifacts.obstacleSilhouettes,
    },
    {
      id: "width-overlay",
      label: "Width Overlay",
      available: !!artifacts.widthOverlay,
    },
    {
      id: "footprint",
      label: "Footprint",
      available: !!artifacts.rectifiedFootprint,
    },
    {
      id: "width-profile",
      label: "Width Profile",
      available: !!artifacts.widthProfile,
    },
  ].filter((t) => t.available);

  return (
    <div>
      <div className="flex gap-1 overflow-x-auto pb-2 mb-3 border-b border-neutral-700/50">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-3 py-1.5 text-xs font-medium rounded-md whitespace-nowrap transition-colors ${
              activeTab === tab.id
                ? "bg-neutral-600 text-white"
                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700/50"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="min-h-[200px]">
        {activeTab === "original" && originalImageUrl && (
          <ImageViewer
            src={originalImageUrl}
            alt="Original streetview image"
            className="w-full aspect-[16/9]"
          />
        )}

        {activeTab === "silhouettes" && artifacts.obstacleSilhouettes && (
          <ImageViewer
            src={artifacts.obstacleSilhouettes}
            alt="Obstacle silhouettes overlay"
            className="w-full aspect-[3/1]"
          />
        )}

        {activeTab === "width-overlay" && artifacts.widthOverlay && (
          <div>
            <ImageViewer
              src={artifacts.widthOverlay}
              alt="Width measurement overlay"
              className="w-full aspect-[3/2]"
            />
            {widthMeta && (
              <div className="mt-3 grid grid-cols-2 gap-2">
                <MetricCard
                  label="Median Width"
                  value={`${widthMeta.width_cm.median.toFixed(1)} cm`}
                />
                <MetricCard
                  label="Mean Width"
                  value={`${widthMeta.width_cm.mean.toFixed(1)} cm`}
                />
                <MetricCard
                  label="Std Dev"
                  value={`${widthMeta.width_cm.std.toFixed(1)} cm`}
                />
                <MetricCard
                  label="Range"
                  value={`${widthMeta.width_cm.min.toFixed(0)}\u2013${widthMeta.width_cm.max.toFixed(0)} cm`}
                />
              </div>
            )}
          </div>
        )}

        {activeTab === "footprint" && artifacts.rectifiedFootprint && (
          <div>
            <ImageViewer
              src={artifacts.rectifiedFootprint}
              alt="Rectified footprint view"
              className="w-full aspect-[2/3]"
            />
            {footprintMeta && footprintMeta.obstacles.length > 0 && (
              <div className="mt-3">
                <h4 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider mb-2">
                  Obstacles
                </h4>
                <div className="space-y-1.5">
                  {footprintMeta.obstacles.map((obs, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between bg-neutral-800/50 rounded-md px-3 py-1.5 text-xs"
                    >
                      <span className="text-neutral-300">{obs.type}</span>
                      <span className="text-neutral-500">
                        {obs.method} &middot; {obs.reduction_pct.toFixed(0)}%
                        reduced
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "width-profile" && artifacts.widthProfile && (
          <ImageViewer
            src={artifacts.widthProfile}
            alt="Width profile chart"
            className="w-full aspect-[5/3]"
          />
        )}
      </div>
    </div>
  );
}

function ReportForm({
  point,
  direction,
}: {
  point: PointData;
  direction: Direction;
}) {
  const [open, setOpen] = useState(false);
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<"success" | "error" | null>(null);

  const handleSubmit = async () => {
    if (!description.trim()) return;
    setSubmitting(true);
    setResult(null);

    try {
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pointId: point.pointId,
          direction,
          latitude: point.latitude,
          longitude: point.longitude,
          description: description.trim(),
        }),
      });
      if (!res.ok) throw new Error("Failed");
      setResult("success");
      setDescription("");
      setTimeout(() => {
        setResult(null);
        setOpen(false);
      }, 2000);
    } catch {
      setResult("error");
    } finally {
      setSubmitting(false);
    }
  };

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 w-full px-3 py-2 mt-4 text-xs font-medium
          rounded-md border border-neutral-700 text-neutral-400
          hover:text-red-400 hover:border-red-500/40 hover:bg-red-500/5 transition-colors"
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
          <line x1="4" y1="22" x2="4" y2="15" />
        </svg>
        Report an issue with this analysis
      </button>
    );
  }

  return (
    <div className="mt-4 rounded-lg border border-neutral-700 bg-neutral-800/40 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-neutral-300">
          Report Issue
        </span>
        <button
          onClick={() => {
            setOpen(false);
            setResult(null);
          }}
          className="text-neutral-500 hover:text-neutral-300 text-sm leading-none"
        >
          &times;
        </button>
      </div>
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Describe the inconsistency or issue..."
        rows={3}
        className="w-full bg-neutral-900 border border-neutral-700 rounded-md px-3 py-2
          text-sm text-neutral-200 placeholder-neutral-500 outline-none resize-none
          focus:border-neutral-500 transition-colors"
      />
      <div className="flex items-center justify-between mt-2">
        {result === "success" && (
          <span className="text-xs text-green-400">Report saved</span>
        )}
        {result === "error" && (
          <span className="text-xs text-red-400">Failed to save</span>
        )}
        {!result && <span />}
        <button
          onClick={handleSubmit}
          disabled={submitting || !description.trim()}
          className="px-3 py-1.5 text-xs font-medium rounded-md transition-colors
            bg-red-600 text-white hover:bg-red-500
            disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {submitting ? "Saving..." : "Submit Report"}
        </button>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-neutral-800/50 rounded-lg px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-neutral-500">
        {label}
      </div>
      <div className="text-sm font-semibold text-neutral-200 mt-0.5">
        {value}
      </div>
    </div>
  );
}

export default function AnalysisPanel({
  open,
  onClose,
  point,
  direction,
  analysis,
  loading,
  onChangeDirection,
}: AnalysisPanelProps) {
  const [activeSegmentIdx, setActiveSegmentIdx] = useState(0);
  const [panelWidth, setPanelWidth] = useState(DEFAULT_WIDTH);
  const isDragging = useRef(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setActiveSegmentIdx(0);
  }, [analysis]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const maxWidth = window.innerWidth * MAX_WIDTH_RATIO;
      const newWidth = Math.max(
        MIN_WIDTH,
        Math.min(maxWidth, window.innerWidth - e.clientX)
      );
      setPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      if (isDragging.current) {
        isDragging.current = false;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  const availableDirections = point
    ? ALL_DIRECTIONS.filter((d) => !!point.directions[d])
    : [];

  return (
    <>
      {/* Backdrop on mobile */}
      {open && (
        <div
          className="fixed inset-0 z-10 bg-black/40 sm:hidden"
          onClick={onClose}
        />
      )}

      <div
        ref={panelRef}
        className={`fixed top-0 right-0 h-full z-20 transition-transform duration-300 ease-in-out
          bg-neutral-900 border-l border-neutral-700/50 shadow-2xl flex
          ${open ? "translate-x-0" : "translate-x-full"}`}
        style={{ width: `min(100vw, ${panelWidth}px)` }}
      >
        {/* Resize handle */}
        <div
          onMouseDown={handleDragStart}
          className="hidden sm:flex w-2 shrink-0 cursor-col-resize items-center justify-center
            hover:bg-neutral-500/20 active:bg-neutral-500/30 transition-colors group"
          title="Drag to resize"
        >
          <div className="w-0.5 h-8 bg-neutral-600 rounded-full group-hover:bg-neutral-400 transition-colors" />
        </div>

        <div className="flex flex-col h-full flex-1 min-w-0">
          {/* Header */}
          <div className="px-4 py-3 border-b border-neutral-700/50">
            <div className="flex items-center justify-between mb-2">
              <div>
                {point && direction && (
                  <>
                    <h2 className="text-base font-semibold text-white">
                      Point {point.pointId}
                    </h2>
                    <p className="text-xs text-neutral-500 mt-0.5">
                      {point.latitude.toFixed(6)},{" "}
                      {point.longitude.toFixed(6)}
                      <span className="mx-1">&middot;</span>
                      Heading:{" "}
                      {point.directions[direction]?.heading.toFixed(1)}
                      &deg;
                    </p>
                  </>
                )}
              </div>
              <button
                onClick={onClose}
                className="text-neutral-400 hover:text-white p-1.5 rounded-md
                  hover:bg-neutral-700/50 transition-colors text-xl leading-none"
              >
                &times;
              </button>
            </div>

            {/* Direction switcher */}
            {point && direction && availableDirections.length > 0 && (
              <div className="flex gap-1">
                {availableDirections.map((dir) => (
                  <button
                    key={dir}
                    onClick={() => {
                      if (dir !== direction) onChangeDirection(dir);
                    }}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md
                      transition-colors ${
                        dir === direction
                          ? "bg-neutral-600 text-white"
                          : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700/50 border border-neutral-700"
                      }`}
                  >
                    <span>{DIRECTION_ARROWS[dir]}</span>
                    <span>{DIRECTION_LABELS[dir]}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto px-4 py-4">
            {loading && (
              <div className="flex flex-col items-center justify-center h-64 gap-3">
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-neutral-400 border-t-transparent" />
                <span className="text-sm text-neutral-400">
                  Loading analysis results...
                </span>
              </div>
            )}

            {!loading && analysis && analysis.segments.length === 0 && (
              <div className="flex flex-col items-center justify-center h-64 gap-2">
                <p className="text-neutral-400 text-sm">
                  No analysis segments found for this direction.
                </p>
                {analysis.originalImageUrl && (
                  <div className="w-full mt-4">
                    <h3 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider mb-2">
                      Original Image
                    </h3>
                    <ImageViewer
                      src={analysis.originalImageUrl}
                      alt="Original streetview image"
                      className="w-full aspect-[16/9]"
                    />
                  </div>
                )}
              </div>
            )}

            {!loading && analysis && analysis.segments.length > 0 && (
              <>
                {analysis.segments.length > 1 && (
                  <div className="flex gap-1 mb-4 overflow-x-auto">
                    {analysis.segments.map((seg, idx) => (
                      <button
                        key={seg.segmentKey}
                        onClick={() => setActiveSegmentIdx(idx)}
                        className={`px-3 py-1.5 text-xs font-medium rounded-md whitespace-nowrap transition-colors ${
                          activeSegmentIdx === idx
                            ? "bg-orange-600 text-white"
                            : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700/50 border border-neutral-700"
                        }`}
                      >
                        {seg.segmentKey}
                      </button>
                    ))}
                  </div>
                )}

                <SegmentView
                  segment={analysis.segments[activeSegmentIdx]}
                  originalImageUrl={analysis.originalImageUrl}
                />
              </>
            )}

            {!loading && !analysis && point && direction && (
              <div className="flex flex-col gap-4">
                <div className="flex items-center gap-2 rounded-md bg-neutral-800/50 px-3 py-2">
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-neutral-500 shrink-0"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                  <p className="text-neutral-400 text-xs">
                    No analysis results available for this point and direction.
                  </p>
                </div>
                {point.directions[direction] && (
                  <div>
                    <h3 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider mb-2">
                      Original Image
                    </h3>
                    <ImageViewer
                      src={`/api/image?blob=${encodeURIComponent(
                        point.directions[direction]!.gcsUri
                      )}`}
                      alt="Original streetview image"
                      className="w-full aspect-[16/9]"
                    />
                  </div>
                )}
              </div>
            )}

            {!loading && point && direction && (
              <ReportForm point={point} direction={direction} />
            )}
          </div>
        </div>
      </div>
    </>
  );
}
