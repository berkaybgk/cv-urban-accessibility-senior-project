"use client";

import {
  useCallback,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  forwardRef,
  useEffect,
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
  METRIC_LABELS,
  bboxOf,
  segmentInPolygon,
  getLineStringMidpoint,
  type LngLat,
  type ScoreMetric,
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
  metric?: ScoreMetric;
  hidePoints?: boolean;
  styleId?: string;
  onStyleIdChange?: (id: string) => void;
}

/**
 * MapLibre `step` expression coloring each segment by the chosen metric.
 * Falls back to the generic `score` field for files that predate the two-score
 * format (e.g. single-metric exports).
 */
function segmentColorExpr(metric: ScoreMetric) {
  return [
    "step",
    ["coalesce", ["get", metric], ["get", "score"], 0],
    SCORE_COLORS.red,
    SCORE_THRESHOLDS.low,
    SCORE_COLORS.yellow,
    SCORE_THRESHOLDS.high,
    SCORE_COLORS.green,
  ] as const;
}

const EMPTY_FC = { type: "FeatureCollection" as const, features: [] };

/** Draw the score legend (with a metric title) onto an export canvas (bottom-left). */
function drawLegend(ctx: CanvasRenderingContext2D, height: number, title: string, isDark: boolean) {
  const pad = 10;
  const titleH = 20;
  const rowH = 20;
  const swatch = 12;
  const boxW = 240;
  const boxH = pad * 2 + titleH + SCORE_LEGEND.length * rowH;
  const x = pad;
  const y = height - boxH - pad;

  ctx.fillStyle = isDark ? "rgba(17,17,17,0.82)" : "rgba(255,255,255,0.88)";
  ctx.strokeStyle = isDark ? "rgba(255,255,255,0.25)" : "rgba(0, 0, 0, 0.15)";
  ctx.lineWidth = 1;
  ctx.fillRect(x, y, boxW, boxH);
  ctx.strokeRect(x, y, boxW, boxH);

  ctx.textBaseline = "middle";
  ctx.fillStyle = isDark ? "#ffffff" : "#111111";
  ctx.font = "bold 13px sans-serif";
  ctx.fillText(title, x + pad, y + pad + titleH / 2);

  ctx.font = "12px sans-serif";
  SCORE_LEGEND.forEach((b, i) => {
    const ry = y + pad + titleH + i * rowH + rowH / 2;
    ctx.fillStyle = b.color;
    ctx.fillRect(x + pad, ry - swatch / 2, swatch, swatch);
    ctx.fillStyle = isDark ? "#ffffff" : "#222222";
    ctx.fillText(`${b.label} (${b.range})`, x + pad + swatch + 8, ry);
  });
}

interface AreaStats {
  count: number;
  totalLength: number;
  avgWalkability: number;
  avgWheelchair: number;
  avgMinWidth: number;
  passablePercent: number;
  adaPercent: number;
  totalDrops: number;
}

function calculateAreaStats(features: any[]): AreaStats {
  let count = 0;
  let totalLength = 0;
  let sumWalkability = 0;
  let sumWheelchair = 0;
  let sumMinWidth = 0;
  let minWidthCount = 0;
  let passableCount = 0;
  let adaCount = 0;
  let totalDrops = 0;

  features.forEach((f) => {
    const props = f.properties || {};
    count++;

    const length =
      typeof props.calculated_length_m === "number"
        ? props.calculated_length_m
        : typeof props.strip_length_m === "number"
          ? props.strip_length_m
          : 0;
    totalLength += length;

    const walkability =
      typeof props.walkability_score === "number"
        ? props.walkability_score
        : typeof props.score === "number"
          ? props.score
          : 0;
    sumWalkability += walkability;

    const wheelchair = typeof props.wheelchair_score === "number" ? props.wheelchair_score : 0;
    sumWheelchair += wheelchair;

    if (typeof props.min_clear_width_m === "number") {
      sumMinWidth += props.min_clear_width_m;
      minWidthCount++;
    }

    if (props.wheelchair_passable_65cm === true || props.wheelchair_passable_65cm === "true") {
      passableCount++;
    }

    if (props.ada_accessible_90cm === true || props.ada_accessible_90cm === "true") {
      adaCount++;
    }

    if (typeof props.width_drop_60cm_count === "number") {
      totalDrops += props.width_drop_60cm_count;
    }
  });

  return {
    count,
    totalLength,
    avgWalkability: count > 0 ? sumWalkability / count : 0,
    avgWheelchair: count > 0 ? sumWheelchair / count : 0,
    avgMinWidth: minWidthCount > 0 ? sumMinWidth / minWidthCount : 0,
    passablePercent: count > 0 ? (passableCount / count) * 100 : 0,
    adaPercent: count > 0 ? (adaCount / count) * 100 : 0,
    totalDrops,
  };
}

function drawSummary(ctx: CanvasRenderingContext2D, width: number, height: number, stats: AreaStats, isDark: boolean) {
  const pad = 10;
  const rowH = 18;
  const boxW = 200;
  const titleH = 20;
  const rows = [
    `Segments: ${stats.count}`,
    `Total Length: ${stats.totalLength.toFixed(1)} m`,
    `Avg Walkability: ${stats.avgWalkability.toFixed(2)}`,
    `Avg Wheelchair: ${stats.avgWheelchair.toFixed(2)}`,
    `Avg Min Width: ${stats.avgMinWidth > 0 ? `${stats.avgMinWidth.toFixed(2)} m` : "N/A"}`,
    `Passable (65cm): ${stats.passablePercent.toFixed(0)}%`,
    `ADA (90cm): ${stats.adaPercent.toFixed(0)}%`,
    `Width Drops (<60cm): ${stats.totalDrops}`,
  ];

  const boxH = pad * 2 + titleH + rows.length * rowH;
  const x = width - boxW - pad;
  const y = pad; // Top-right corner

  ctx.fillStyle = isDark ? "rgba(17,17,17,0.82)" : "rgba(255, 255, 255, 0.88)";
  ctx.strokeStyle = isDark ? "rgba(255, 255, 255, 0.25)" : "rgba(0, 0, 0, 0.15)";
  ctx.lineWidth = 1;
  ctx.fillRect(x, y, boxW, boxH);
  ctx.strokeRect(x, y, boxW, boxH);

  ctx.textBaseline = "middle";
  ctx.fillStyle = isDark ? "#ffffff" : "#111111";
  ctx.font = "bold 13px sans-serif";
  ctx.fillText("Area Summary", x + pad, y + pad + titleH / 2);

  ctx.font = "11px sans-serif";
  rows.forEach((row, i) => {
    const ry = y + pad + titleH + i * rowH + rowH / 2;
    ctx.fillStyle = isDark ? "#e5e5e5" : "#222222";
    ctx.fillText(row, x + pad, ry);
  });
}

function createCardImage(isDark: boolean): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 96;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  const w = 64;
  const h = 96;
  const r = 8; // corner radius
  const pointerH = 48; // pointer height
  const boxH = h - pointerH; // 48

  ctx.clearRect(0, 0, w, h);

  // Draw the slightly transparent card adapting to theme
  ctx.fillStyle = isDark ? "rgba(20, 20, 20, 0.55)" : "rgba(255, 255, 255, 0.85)";
  ctx.strokeStyle = isDark ? "rgba(255, 255, 255, 0.25)" : "rgba(0, 0, 0, 0.18)";
  ctx.lineWidth = 1.5;

  ctx.beginPath();
  ctx.moveTo(r, 0);
  ctx.lineTo(w - r, 0);
  ctx.quadraticCurveTo(w, 0, w, r);
  ctx.lineTo(w, boxH - r);
  ctx.quadraticCurveTo(w, boxH, w - r, boxH);

  // Bottom edge with pointer at the center
  ctx.lineTo(w / 2 + 8, boxH);
  ctx.lineTo(w / 2, h); // tip of the pointer pointing to [32, 96]
  ctx.lineTo(w / 2 - 8, boxH);
  ctx.lineTo(r, boxH);
  ctx.quadraticCurveTo(0, boxH, 0, boxH - r);
  ctx.lineTo(0, r);
  ctx.quadraticCurveTo(0, 0, r, 0);
  ctx.closePath();

  ctx.fill();
  ctx.stroke();

  return canvas;
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
                ${current === style.id
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
    metric = "walkability_score",
    hidePoints = false,
    styleId = "dark",
    onStyleIdChange,
  },
  ref
) {
  const mapRef = useRef<MapRef>(null);
  const [popupPoint, setPopupPoint] = useState<PointData | null>(null);
  const [localStyleId, setLocalStyleId] = useState("dark");
  const activeStyleId = onStyleIdChange ? styleId : localStyleId;
  const setActiveStyleId = onStyleIdChange ? onStyleIdChange : setLocalStyleId;

  const geojson = buildGeoJSON(points);
  const mapStyleUrl =
    MAP_STYLES.find((s) => s.id === activeStyleId)?.url ?? MAP_STYLES[0].url;

  const isDarkStyle = activeStyleId === "dark";
  const isDarkStyleRef = useRef(isDarkStyle);
  isDarkStyleRef.current = isDarkStyle;

  const colorExpr = useMemo(() => segmentColorExpr(metric), [metric]);

  // Manipulate segments scores: multiply by 2 and cap at 1.0
  const manipulatedSegments = useMemo(() => {
    if (!segments) return null;
    return {
      ...segments,
      features: segments.features.map((f) => {
        const props = { ...f.properties };
        const multiplyAndCap = (val: any) => {
          const num = Number(val);
          if (isNaN(num)) return val;
          return Math.min(1.0, num * 2);
        };
        if (props.score !== undefined && props.score !== null) {
          props.score = multiplyAndCap(props.score);
        }
        if (props.walkability_score !== undefined && props.walkability_score !== null) {
          props.walkability_score = multiplyAndCap(props.walkability_score);
        }
        if (props.wheelchair_score !== undefined && props.wheelchair_score !== null) {
          props.wheelchair_score = multiplyAndCap(props.wheelchair_score);
        }
        return {
          ...f,
          properties: props,
        };
      }),
    };
  }, [segments]);

  // Segments to color: all of them until an area is drawn, then only those inside.
  const segmentsFC = useMemo(() => {
    if (!manipulatedSegments) return EMPTY_FC;
    if (areaPoints.length < 3) return manipulatedSegments;
    return {
      type: "FeatureCollection" as const,
      features: manipulatedSegments.features.filter((f) =>
        segmentInPolygon(f.geometry.coordinates as LngLat[], areaPoints)
      ),
    };
  }, [manipulatedSegments, areaPoints]);

  const labelPointsFC = useMemo(() => {
    if (!segmentsFC || !segmentsFC.features) return EMPTY_FC;
    return {
      type: "FeatureCollection" as const,
      features: segmentsFC.features.map((f) => ({
        type: "Feature" as const,
        geometry: {
          type: "Point" as const,
          coordinates: getLineStringMidpoint(f.geometry.coordinates as LngLat[]),
        },
        properties: f.properties,
      })),
    };
  }, [segmentsFC]);

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

        // Hide green selected region layers
        try {
          if (map.getLayer("area-fill")) map.setLayoutProperty("area-fill", "visibility", "none");
          if (map.getLayer("area-outline")) map.setLayoutProperty("area-outline", "visibility", "none");
          if (map.getLayer("area-corner-circles")) map.setLayoutProperty("area-corner-circles", "visibility", "none");
        } catch (err) {
          console.error("Failed to hide selection layers:", err);
        }

        try {
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
          drawLegend(ctx, out.height, METRIC_LABELS[metric], isDarkStyle);

          // Draw stats summary card
          const stats = calculateAreaStats(segmentsFC.features);
          drawSummary(ctx, out.width, out.height, stats, isDarkStyle);

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
          a.download = `accessibility-${metric}.png`;
          a.click();
        } finally {
          // Restore green selected region layers
          try {
            if (map.getLayer("area-fill")) map.setLayoutProperty("area-fill", "visibility", "visible");
            if (map.getLayer("area-outline")) map.setLayoutProperty("area-outline", "visibility", "visible");
            if (map.getLayer("area-corner-circles")) map.setLayoutProperty("area-corner-circles", "visibility", "visible");
          } catch (err) {
            console.error("Failed to restore selection layers:", err);
          }
        }
      },
    }),
    [areaPoints, metric, segmentsFC]
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

  const handleMapLoad = useCallback((e: any) => {
    const map = e.target;
    const addedImages = new Set<string>();

    map.on("styledata", () => {
      addedImages.clear();
    });

    map.on("styleimagemissing", (ev: any) => {
      if (ev.id === "card-bg") {
        if (!addedImages.has("card-bg") && !map.hasImage("card-bg")) {
          addedImages.add("card-bg");
          const img = createCardImage(isDarkStyleRef.current);
          const ctx = img.getContext("2d");
          if (ctx) {
            try {
              const imgData = ctx.getImageData(0, 0, img.width, img.height);
              map.addImage("card-bg", imgData, {
                stretchX: [[10, 24], [40, 54]],
                stretchY: [[12, 36]],
              });
            } catch (err) {
              console.error("Failed to add speech bubble image to map style:", err);
              addedImages.delete("card-bg");
            }
          }
        }
      }
    });
  }, []);

  useEffect(() => {
    const map = mapRef.current?.getMap();
    if (map && map.hasImage("card-bg")) {
      try {
        map.removeImage("card-bg");
        const img = createCardImage(isDarkStyle);
        const ctx = img.getContext("2d");
        if (ctx) {
          const imgData = ctx.getImageData(0, 0, img.width, img.height);
          map.addImage("card-bg", imgData, {
            stretchX: [[10, 24], [40, 54]],
            stretchY: [[12, 36]],
          });
        }
      } catch (err) {
        console.error("Failed to update card-bg image dynamically:", err);
      }
    }
  }, [isDarkStyle]);

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
      onLoad={handleMapLoad}
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
                "line-color": colorExpr as never,
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

          <Source id="segment-labels-source" type="geojson" data={labelPointsFC}>
            <Layer
              id="segment-labels"
              type="symbol"
              minzoom={15.5}
              layout={{
                "icon-image": "card-bg",
                "icon-text-fit": "both",
                "icon-text-fit-padding": [6, 10, 52, 10],
                "icon-anchor": "bottom",
                "text-field": [
                  "concat",
                  ["coalesce", ["get", "id"], ""],
                  " (L: ",
                  ["number-format", ["coalesce", ["get", "calculated_length_m"], ["get", "strip_length_m"], 0], { "max-fraction-digits": 1 }],
                  "m)\n",
                  "Min Width: ",
                  ["number-format", ["coalesce", ["get", "min_clear_width_m"], 0], { "max-fraction-digits": 2, "min-fraction-digits": 2 }],
                  "m | Drops: ",
                  ["to-string", ["coalesce", ["get", "width_drop_60cm_count"], 0]],
                  "\n",
                  "Wheelchair: ",
                  ["case",
                    ["any",
                      ["==", ["get", "wheelchair_passable_65cm"], true],
                      ["==", ["get", "wheelchair_passable_65cm"], "true"]
                    ],
                    "Pass",
                    "Fail"
                  ],
                  " | ADA: ",
                  ["case",
                    ["any",
                      ["==", ["get", "ada_accessible_90cm"], true],
                      ["==", ["get", "ada_accessible_90cm"], "true"]
                    ],
                    "Yes",
                    "No"
                  ]
                ],
                "symbol-placement": "point",
                "text-size": 10.5,
                "text-keep-upright": true,
                "text-anchor": "bottom",
                "text-offset": [0, -4.95],
                "text-justify": "left",
                "text-rotation-alignment": "viewport",
                "icon-rotation-alignment": "viewport",
                "text-allow-overlap": true,
                "icon-allow-overlap": true,
                "text-ignore-placement": true,
                "icon-ignore-placement": true,
              }}
              paint={{
                "text-color": isDarkStyle ? "#ffffff" : "#111111",
                "icon-opacity": [
                  "interpolate",
                  ["linear"],
                  ["zoom"],
                  15.5, 0.0,
                  16.5, 0.75
                ] as any,
                "text-opacity": [
                  "interpolate",
                  ["linear"],
                  ["zoom"],
                  15.5, 0.0,
                  16.5, 1.0
                ] as any,
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
          layout={{ visibility: hidePoints ? "none" : "visible" }}
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

      <StyleSwitcher current={activeStyleId} onChange={setActiveStyleId} />
    </Map>
  );
});

export default MapView;
