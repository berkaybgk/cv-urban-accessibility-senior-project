"use client";

import { useCallback, useImperativeHandle, useRef, useState, forwardRef } from "react";
import Map, {
  Layer,
  Source,
  Popup,
  MapRef,
  MapLayerMouseEvent,
} from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import type { PointData, PointsHashMap, Direction } from "@/lib/types";

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

interface MapStyle {
  id: string;
  label: string;
  url: string;
}

const MAP_STYLES: MapStyle[] = [
  {
    id: "dark",
    label: "Dark",
    url: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  },
  {
    id: "light",
    label: "Light",
    url: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  },
  {
    id: "voyager",
    label: "Voyager",
    url: "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
  },
  {
    id: "osm",
    label: "OpenStreetMap",
    url: "https://tiles.openfreemap.org/styles/liberty",
  },
];

export interface MapViewHandle {
  flyToPoint: (point: PointData) => void;
}

interface MapViewProps {
  points: PointsHashMap;
  onSelectDirection: (point: PointData, direction: Direction) => void;
  selectedPointId: string | null;
}

function buildGeoJSON(points: PointsHashMap) {
  const features = Object.values(points).map((pt) => ({
    type: "Feature" as const,
    geometry: {
      type: "Point" as const,
      coordinates: [pt.longitude, pt.latitude],
    },
    properties: {
      pointId: pt.pointId,
      bearing: pt.streetBearing,
    },
  }));

  return { type: "FeatureCollection" as const, features };
}

function StyleSwitcher({
  current,
  onChange,
}: {
  current: string;
  onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="absolute bottom-6 left-4 z-10">
      {open && (
        <div className="mb-2 bg-neutral-900/90 backdrop-blur-sm rounded-lg border border-neutral-700/50 overflow-hidden">
          {MAP_STYLES.map((style) => (
            <button
              key={style.id}
              onClick={() => {
                onChange(style.id);
                setOpen(false);
              }}
              className={`block w-full text-left px-4 py-2 text-xs font-medium transition-colors
                ${
                  current === style.id
                    ? "bg-neutral-600/40 text-white"
                    : "text-neutral-300 hover:bg-neutral-700/50 hover:text-white"
                }`}
            >
              {style.label}
            </button>
          ))}
        </div>
      )}
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 bg-neutral-900/80 backdrop-blur-sm
          rounded-lg px-3 py-2 border border-neutral-700/50
          text-xs font-medium text-neutral-300 hover:text-white transition-colors"
        title="Switch map style"
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
          <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6" />
          <line x1="8" y1="2" x2="8" y2="18" />
          <line x1="16" y1="6" x2="16" y2="22" />
        </svg>
        <span>{MAP_STYLES.find((s) => s.id === current)?.label ?? "Map"}</span>
      </button>
    </div>
  );
}

const MapView = forwardRef<MapViewHandle, MapViewProps>(function MapView(
  { points, onSelectDirection, selectedPointId },
  ref
) {
  const mapRef = useRef<MapRef>(null);
  const [popupPoint, setPopupPoint] = useState<PointData | null>(null);
  const [styleId, setStyleId] = useState("dark");

  const geojson = buildGeoJSON(points);
  const mapStyleUrl =
    MAP_STYLES.find((s) => s.id === styleId)?.url ?? MAP_STYLES[0].url;

  const isDarkStyle = styleId === "dark";

  useImperativeHandle(ref, () => ({
    flyToPoint(point: PointData) {
      const map = mapRef.current;
      if (map) {
        map.flyTo({
          center: [point.longitude, point.latitude],
          zoom: 17,
          duration: 1200,
        });
      }
      setPopupPoint(point);
    },
  }));

  const handleClick = useCallback(
    (e: MapLayerMouseEvent) => {
      const feature = e.features?.[0];
      if (!feature || !feature.properties) {
        setPopupPoint(null);
        return;
      }
      const pid = feature.properties.pointId as string;
      const pt = points[pid];
      if (pt) setPopupPoint(pt);
    },
    [points]
  );

  const handleMouseEnter = useCallback(() => {
    const map = mapRef.current?.getMap();
    if (map) map.getCanvas().style.cursor = "pointer";
  }, []);

  const handleMouseLeave = useCallback(() => {
    const map = mapRef.current?.getMap();
    if (map) map.getCanvas().style.cursor = "";
  }, []);

  return (
    <Map
      ref={mapRef}
      initialViewState={{
        longitude: 29.06,
        latitude: 40.97,
        zoom: 14,
      }}
      style={{ width: "100%", height: "100%" }}
      mapStyle={mapStyleUrl}
      interactiveLayerIds={["point-circles"]}
      onClick={handleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <Source id="points" type="geojson" data={geojson}>
        <Layer
          id="point-circles"
          type="circle"
          paint={{
            "circle-radius": [
              "interpolate",
              ["linear"],
              ["zoom"],
              12, 3,
              15, 6,
              18, 10,
            ],
            "circle-color": [
              "case",
              ["==", ["get", "pointId"], selectedPointId ?? ""],
              "#f97316",
              "#a3a3a3",
            ],
            "circle-stroke-width": 1.5,
            "circle-stroke-color": [
              "case",
              ["==", ["get", "pointId"], selectedPointId ?? ""],
              "#ffffff",
              isDarkStyle ? "#525252" : "#e5e5e5",
            ],
            "circle-opacity": 0.9,
          }}
        />
      </Source>

      {popupPoint && (
        <Popup
          longitude={popupPoint.longitude}
          latitude={popupPoint.latitude}
          anchor="bottom"
          onClose={() => setPopupPoint(null)}
          closeOnClick={false}
          className="point-popup"
          maxWidth="280px"
        >
          <div className="p-1">
            <div className="text-sm font-semibold text-neutral-900 mb-1">
              Point {popupPoint.pointId}
            </div>
            <div className="text-xs text-neutral-500 mb-2">
              {popupPoint.latitude.toFixed(6)},{" "}
              {popupPoint.longitude.toFixed(6)}
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              {(Object.keys(DIRECTION_LABELS) as Direction[]).map((dir) => {
                const dirData = popupPoint.directions[dir];
                if (!dirData) return null;
                return (
                  <button
                    key={dir}
                    onClick={() => {
                      onSelectDirection(popupPoint, dir);
                      setPopupPoint(null);
                    }}
                    className="flex items-center justify-center gap-1.5 px-2 py-1.5
                      text-xs font-medium rounded-md transition-colors
                      bg-neutral-100 text-neutral-700 hover:bg-neutral-200
                      border border-neutral-300 hover:border-neutral-400"
                  >
                    <span>{DIRECTION_ARROWS[dir]}</span>
                    <span>{DIRECTION_LABELS[dir]}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </Popup>
      )}

      <StyleSwitcher current={styleId} onChange={setStyleId} />
    </Map>
  );
});

export default MapView;
