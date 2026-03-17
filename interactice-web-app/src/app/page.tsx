"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type {
  PointData,
  PointsHashMap,
  Direction,
  AnalysisResult,
} from "@/lib/types";
import type { MapViewHandle } from "@/components/MapView";
import AnalysisPanel from "@/components/AnalysisPanel";
import PointSearch from "@/components/PointSearch";

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

export default function HomePage() {
  const [points, setPoints] = useState<PointsHashMap>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedPoint, setSelectedPoint] = useState<PointData | null>(null);
  const [selectedDirection, setSelectedDirection] = useState<Direction | null>(
    null
  );
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);

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

      try {
        const data = await fetchAnalysis(point, direction);
        setAnalysis(data);
      } catch (err) {
        console.error(err);
        setAnalysis(null);
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
  }, []);

  const handleSearchSelect = useCallback((point: PointData) => {
    mapViewRef.current?.flyToPoint(point);
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
      </div>

      <AnalysisPanel
        open={panelOpen}
        onClose={handleClosePanel}
        point={selectedPoint}
        direction={selectedDirection}
        analysis={analysis}
        loading={analysisLoading}
        onChangeDirection={handleChangeDirection}
      />
    </div>
  );
}
