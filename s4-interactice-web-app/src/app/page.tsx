"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type {
  PointData,
  PointsHashMap,
  Direction,
  AnalysisResult,
  AlternativeWidthResult,
} from "@/lib/types";
import type { MapViewHandle } from "@/components/MapView";
import AnalysisPanel from "@/components/AnalysisPanel";
import PointSearch from "@/components/PointSearch";
import StripBuilder from "@/components/StripBuilder";

const MapView = dynamic(() => import("@/components/MapView"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center w-full h-full bg-neutral-900">
      <div className="flex flex-col items-center gap-3">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-neutral-400 border-t-transparent" />
        <span className="text-sm text-neutral-400">Loading map...</span>
      </div>
    </div>
  ),
});

async function fetchAnalysis(
  point: PointData,
  direction: Direction
): Promise<AnalysisResult> {
  const dirData = point.directions[direction];
  const params = new URLSearchParams({
    pointId: point.pointId,
    direction,
    lat: String(point.latitude),
    lon: String(point.longitude),
    ...(dirData ? { originalBlob: dirData.gcsUri } : {}),
  });

  const res = await fetch(`/api/analysis?${params}`);
  if (!res.ok) throw new Error("Failed to fetch analysis");
  return res.json();
}

async function fetchAlternativeWidth(
  point: PointData,
  direction: Direction
): Promise<AlternativeWidthResult> {
  const params = new URLSearchParams({
    pointId: point.pointId,
    direction,
    lat: String(point.latitude),
    lon: String(point.longitude),
  });
  const res = await fetch(`/api/alt-width?${params}`);
  if (!res.ok) throw new Error("Failed to fetch alternative width");
  return res.json();
}

export default function HomePage() {
  const [points, setPoints] = useState<PointsHashMap>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedPoint, setSelectedPoint] = useState<PointData | null>(null);
  const [selectedDirection, setSelectedDirection] = useState<Direction | null>(
    null
  );
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [alternativeWidth, setAlternativeWidth] =
    useState<AlternativeWidthResult | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);

  const [stripMode, setStripMode] = useState(false);
  const [stripSelectedIds, setStripSelectedIds] = useState<string[]>([]);
  const [stripBuilderOpen, setStripBuilderOpen] = useState(false);

  const mapViewRef = useRef<MapViewHandle>(null);

  useEffect(() => {
    fetch("/api/points")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load points");
        return res.json();
      })
      .then((data) => {
        setPoints(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const loadAnalysis = useCallback(
    async (point: PointData, direction: Direction) => {
      setSelectedPoint(point);
      setSelectedDirection(direction);
      setPanelOpen(true);
      setAnalysisLoading(true);
      setAnalysis(null);
      setAlternativeWidth(null);

      try {
        const [analysisData, altData] = await Promise.all([
          fetchAnalysis(point, direction),
          fetchAlternativeWidth(point, direction),
        ]);
        setAnalysis(analysisData);
        setAlternativeWidth(altData);
      } catch (err) {
        console.error(err);
        setAnalysis(null);
        setAlternativeWidth(null);
      } finally {
        setAnalysisLoading(false);
      }
    },
    []
  );

  const handleSelectDirection = useCallback(
    (point: PointData, direction: Direction) => {
      loadAnalysis(point, direction);
    },
    [loadAnalysis]
  );

  const handleChangeDirection = useCallback(
    (direction: Direction) => {
      if (!selectedPoint) return;
      loadAnalysis(selectedPoint, direction);
    },
    [selectedPoint, loadAnalysis]
  );

  const handleClosePanel = useCallback(() => {
    setPanelOpen(false);
    setSelectedPoint(null);
    setSelectedDirection(null);
    setAnalysis(null);
    setAlternativeWidth(null);
  }, []);

  const handleSearchSelect = useCallback((point: PointData) => {
    mapViewRef.current?.flyToPoint(point);
  }, []);

  const handleToggleStripPoint = useCallback((pointId: string) => {
    setStripSelectedIds((prev) =>
      prev.includes(pointId) ? prev.filter((id) => id !== pointId) : [...prev, pointId]
    );
  }, []);

  const toggleStripMode = useCallback(() => {
    setStripMode((on) => {
      const next = !on;
      if (next) {
        setPanelOpen(false);
      }
      return next;
    });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-neutral-900">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-neutral-400 border-t-transparent" />
          <span className="text-neutral-300">Loading coordinate data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-neutral-900">
        <div className="text-center">
          <p className="text-red-400 text-lg font-medium mb-2">Error</p>
          <p className="text-neutral-400">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      <MapView
        ref={mapViewRef}
        points={points}
        onSelectDirection={handleSelectDirection}
        selectedPointId={selectedPoint?.pointId ?? null}
        stripMode={stripMode}
        stripSelectedIds={stripSelectedIds}
        onToggleStripPoint={handleToggleStripPoint}
      />

      {/* Top bar: title + search */}
      <div className="absolute top-4 left-4 z-10 flex items-start gap-3">
        <div
          className="bg-neutral-900/80 backdrop-blur-sm
          rounded-lg px-4 py-2 border border-neutral-700/50"
        >
          <h1 className="text-sm font-semibold text-white">
            Sidewalk Analysis Viewer
          </h1>
          <p className="text-xs text-neutral-400">
            {Object.keys(points).length} points &middot; Click a dot to explore
          </p>
        </div>

        <PointSearch
          points={points}
          onSelectPoint={handleSearchSelect}
          onSelectDirection={handleSelectDirection}
        />

        <button
          onClick={toggleStripMode}
          className={`rounded-lg border px-4 py-2 text-sm font-medium backdrop-blur-sm transition-colors ${
            stripMode
              ? "border-cyan-500/60 bg-cyan-600/80 text-white"
              : "border-neutral-700/50 bg-neutral-900/80 text-neutral-200 hover:text-white"
          }`}
          title="Pick points to merge into a continuous strip"
        >
          {stripMode ? "Strip mode: ON" : "Strip mode"}
        </button>
      </div>

      {/* Strip selection panel */}
      {stripMode && (
        <div className="absolute right-4 top-4 z-20 w-64 rounded-lg border border-neutral-700/60 bg-neutral-900/90 p-3 text-neutral-100 backdrop-blur-sm">
          <div className="mb-2 text-sm font-semibold">Strip points</div>
          <p className="mb-2 text-xs text-neutral-400">
            Click map dots to add/remove. Order is auto-derived from street direction.
          </p>
          {stripSelectedIds.length === 0 ? (
            <p className="text-xs text-neutral-500">No points selected yet.</p>
          ) : (
            <div className="mb-3 flex flex-wrap gap-1.5">
              {stripSelectedIds.map((id) => (
                <button
                  key={id}
                  onClick={() => handleToggleStripPoint(id)}
                  className="rounded-md bg-cyan-700/60 px-2 py-1 text-xs hover:bg-red-700/60"
                  title="Remove"
                >
                  {id} ✕
                </button>
              ))}
            </div>
          )}
          <div className="flex gap-2">
            <button
              onClick={() => setStripBuilderOpen(true)}
              disabled={stripSelectedIds.length < 2}
              className="flex-1 rounded-md bg-cyan-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-cyan-500 disabled:opacity-40"
            >
              Merge them into a strip
            </button>
            {stripSelectedIds.length > 0 && (
              <button
                onClick={() => setStripSelectedIds([])}
                className="rounded-md border border-neutral-700 px-2 py-1.5 text-xs hover:bg-neutral-800"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      )}

      <StripBuilder
        open={stripBuilderOpen}
        pointIds={stripSelectedIds}
        onClose={() => setStripBuilderOpen(false)}
      />

      <AnalysisPanel
        open={panelOpen}
        onClose={handleClosePanel}
        point={selectedPoint}
        direction={selectedDirection}
        analysis={analysis}
        alternativeWidth={alternativeWidth}
        loading={analysisLoading}
        onChangeDirection={handleChangeDirection}
      />
    </div>
  );
}
