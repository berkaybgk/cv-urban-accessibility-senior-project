"use client";

import {
  useCallback,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  forwardRef,
} from "react";
import Map, {
  Layer,
  Source,
  Popup,
  MapRef,
  MapLayerMouseEvent,
} from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import type {
  PointData,
  PointsHashMap,
  Direction,
  ScoredSegmentCollection,
} from "@/lib/types";
import {
  SCORE_COLORS,
  SCORE_THRESHOLDS,
  SCORE_LEGEND,
  bboxOf,
  segmentInPolygon,
  type LngLat,
} from "@/lib/geo";

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
  exportAreaPng: () => Promise<void>;
}

interface MapViewProps {
  points: PointsHashMap;
  onSelectDirection: (point: PointData, direction: Direction) => void;
  selectedPointId: string | null;
  stripMode?: boolean;
  stripSelectedIds?: string[];
  onToggleStripPoint?: (pointId: string) => void;
  accessMode?: boolean;
  areaPoints?: LngLat[];
  segments?: ScoredSegmentCollection | null;
  onAddAreaPoint?: (lngLat: LngLat) => void;
}

const SEGMENT_COLOR_EXPR = [
  "step",
  ["get", "score"],
  SCORE_COLORS.red,
  SCORE_THRESHOLDS.low,
  SCORE_COLORS.yellow,
  SCORE_THRESHOLDS.high,
  SCORE_COLORS.green,
] as const;

const EMPTY_FC = { type: "FeatureCollection" as const, features: [] };

/** Draw the score legend onto an export canvas (bottom-left). */
function drawLegend(ctx: CanvasRenderingContext2D, height: number) {
  const pad = 10;
  const rowH = 20;
  const swatch = 12;
  const boxW = 168;
  const boxH = pad * 2 + SCORE_LEGEND.length * rowH;
  const x = pad;
  const y = height - boxH - pad;

  ctx.fillStyle = "rgba(17,17,17,0.82)";
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.lineWidth = 1;
  ctx.fillRect(x, y, boxW, boxH);
  ctx.strokeRect(x, y, boxW, boxH);

  ctx.font = "12px sans-serif";
  ctx.textBaseline = "middle";
  SCORE_LEGEND.forEach((b, i) => {
    const ry = y + pad + i * rowH + rowH / 2;
    ctx.fillStyle = b.color;
    ctx.fillRect(x + pad, ry - swatch / 2, swatch, swatch);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(`${b.label}  (${b.range})`, x + pad + swatch + 8, ry);
  });
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
  {
    points,
    onSelectDirection,
    selectedPointId,
    stripMode = false,
    stripSelectedIds = [],
    onToggleStripPoint,
    accessMode = false,
    areaPoints = [],
    segments = null,
    onAddAreaPoint,
  },
  ref
) {
  const mapRef = useRef<MapRef>(null);
  const [popupPoint, setPopupPoint] = useState<PointData | null>(null);
  const [styleId, setStyleId] = useState("dark");

  const geojson = buildGeoJSON(points);
  const mapStyleUrl =
    MAP_STYLES.find((s) => s.id === styleId)?.url ?? MAP_STYLES[0].url;

  const isDarkStyle = styleId === "dark";

  // Segments to color: all of them until an area is drawn, then only those inside.
  const segmentsFC = useMemo(() => {
    if (!segments) return EMPTY_FC;
    if (areaPoints.length < 3) return segments;
    return {
      type: "FeatureCollection" as const,
      features: segments.features.filter((f) =>
        segmentInPolygon(f.geometry.coordinates as LngLat[], areaPoints)
      ),
    };
  }, [segments, areaPoints]);

  const areaPolygonFC = useMemo(() => {
    if (areaPoints.length < 3) return EMPTY_FC;
    return {
      type: "FeatureCollection" as const,
      features: [
        {
          type: "Feature" as const,
          properties: {},
          geometry: {
            type: "Polygon" as const,
            coordinates: [[...areaPoints, areaPoints[0]]],
          },
        },
      ],
    };
  }, [areaPoints]);

  const areaCornersFC = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: areaPoints.map((c) => ({
        type: "Feature" as const,
        properties: {},
        geometry: { type: "Point" as const, coordinates: c },
      })),
    }),
    [areaPoints]
  );

  useImperativeHandle(
    ref,
    () => ({
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
      async exportAreaPng() {
        const map = mapRef.current?.getMap();
        if (!map) throw new Error("Map is not ready.");
        if (areaPoints.length !== 4) {
          throw new Error("Pick 4 corners on the map first.");
        }
        const [minLng, minLat, maxLng, maxLat] = bboxOf(areaPoints);
        map.fitBounds(
          [
            [minLng, minLat],
            [maxLng, maxLat],
          ],
          { padding: 48, animate: false }
        );
        await new Promise<void>((resolve) => map.once("idle", () => resolve()));

        const canvas = map.getCanvas();
        const scaleX = canvas.width / canvas.clientWidth;
        const scaleY = canvas.height / canvas.clientHeight;
        const projected = areaPoints.map((c) => map.project(c));
        const xs = projected.map((p) => p.x);
        const ys = projected.map((p) => p.y);
        const sx = Math.max(0, Math.floor(Math.min(...xs) * scaleX));
        const sy = Math.max(0, Math.floor(Math.min(...ys) * scaleY));
        const sw = Math.min(
          canvas.width - sx,
          Math.ceil((Math.max(...xs) - Math.min(...xs)) * scaleX)
        );
        const sh = Math.min(
          canvas.height - sy,
          Math.ceil((Math.max(...ys) - Math.min(...ys)) * scaleY)
        );

        const out = document.createElement("canvas");
        out.width = Math.max(1, sw);
        out.height = Math.max(1, sh);
        const ctx = out.getContext("2d");
        if (!ctx) throw new Error("Could not create export canvas.");
        ctx.drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh);
        drawLegend(ctx, out.height);

        let url: string;
        try {
          url = out.toDataURL("image/png");
        } catch {
          throw new Error(
            "Export blocked by the basemap (CORS). Switch to the Light or Voyager style and retry."
          );
        }
        const a = document.createElement("a");
        a.href = url;
        a.download = "accessibility-area.png";
        a.click();
      },
    }),
    [areaPoints]
  );

  const handleClick = useCallback(
    (e: MapLayerMouseEvent) => {
      const feature = e.features?.[0];
      if (accessMode) {
        if (!feature) onAddAreaPoint?.([e.lngLat.lng, e.lngLat.lat]);
        return;
      }
      if (!feature || !feature.properties) {
        if (!stripMode) setPopupPoint(null);
        return;
      }
      const pid = feature.properties.pointId as string;
      if (stripMode) {
        onToggleStripPoint?.(pid);
        return;
      }
      const pt = points[pid];
      if (pt) setPopupPoint(pt);
    },
    [points, stripMode, accessMode, onAddAreaPoint, onToggleStripPoint]
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
      preserveDrawingBuffer
    >
      {accessMode && (
        <>
          <Source id="area-polygon" type="geojson" data={areaPolygonFC}>
            <Layer
              id="area-fill"
              type="fill"
              paint={{ "fill-color": "#10b981", "fill-opacity": 0.08 }}
            />
            <Layer
              id="area-outline"
              type="line"
              paint={{
                "line-color": "#10b981",
                "line-width": 2,
                "line-dasharray": [2, 1],
              }}
            />
          </Source>

          <Source id="segments" type="geojson" data={segmentsFC}>
            <Layer
              id="segment-lines"
              type="line"
              layout={{ "line-cap": "round", "line-join": "round" }}
              paint={{
                "line-color": SEGMENT_COLOR_EXPR as never,
                "line-width": [
                  "interpolate",
                  ["linear"],
                  ["zoom"],
                  12, 3,
                  16, 6,
                  19, 10,
                ],
                "line-opacity": 0.95,
              }}
            />
          </Source>

          <Source id="area-corners" type="geojson" data={areaCornersFC}>
            <Layer
              id="area-corner-circles"
              type="circle"
              paint={{
                "circle-radius": 6,
                "circle-color": "#10b981",
                "circle-stroke-width": 2,
                "circle-stroke-color": "#ffffff",
              }}
            />
          </Source>
        </>
      )}

      <Source id="points" type="geojson" data={geojson}>
        <Layer
          id="point-circles"
          type="circle"
          paint={{
            "circle-radius": [
              "interpolate",
              ["linear"],
              ["zoom"],
              12, ["case", ["in", ["get", "pointId"], ["literal", stripSelectedIds]], 5, 3],
              15, ["case", ["in", ["get", "pointId"], ["literal", stripSelectedIds]], 9, 6],
              18, ["case", ["in", ["get", "pointId"], ["literal", stripSelectedIds]], 13, 10],
            ],
            "circle-color": [
              "case",
              ["in", ["get", "pointId"], ["literal", stripSelectedIds]],
              "#22d3ee",
              ["==", ["get", "pointId"], selectedPointId ?? ""],
              "#a855f7",
              "#f97316",
            ],
            "circle-stroke-width": [
              "case",
              ["in", ["get", "pointId"], ["literal", stripSelectedIds]],
              2.5,
              1.5,
            ],
            "circle-stroke-color": [
              "case",
              ["in", ["get", "pointId"], ["literal", stripSelectedIds]],
              "#cffafe",
              ["==", ["get", "pointId"], selectedPointId ?? ""],
              "#e9d5ff",
              isDarkStyle ? "#7c2d12" : "#fed7aa",
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
