export type Direction = "forward" | "right" | "backward" | "left";

export const ALL_DIRECTIONS: Direction[] = [
  "forward",
  "right",
  "backward",
  "left",
];

export interface DirectionData {
  gcsUri: string;
  heading: number;
}

export interface PointData {
  pointId: string;
  latitude: number;
  longitude: number;
  streetBearing: number;
  panoId: string;
  directions: Partial<Record<Direction, DirectionData>>;
}

export type PointsHashMap = Record<string, PointData>;

export interface AnalysisSegment {
  segmentKey: string;
  artifacts: {
    obstacleSilhouettes?: string;
    rectifiedFootprint?: string;
    widthOverlay?: string;
    widthProfile?: string;
    footprintMetadata?: string;
    widthMetadata?: string;
  };
}

export interface AnalysisResult {
  pointId: string;
  direction: Direction;
  coordinateFolder: string;
  originalImageUrl: string;
  segments: AnalysisSegment[];
}

export interface AlternativeWidthResult {
  pointId: string;
  direction: Direction;
  coordinateFolder: string;
  metadataUrl?: string;
  overlayUrl?: string;
  scene3dUrl?: string;
  width3dUrl?: string;
}

export interface WidthMetadata {
  segment: string;
  width_cm: {
    median: number;
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  rows_used: number;
}

export interface FootprintMetadata {
  segment: string;
  rectified_height: number;
  rectified_width: number;
  sidewalk_target_width_px: number;
  padding_px: number;
  sidewalk_area_px: number;
  obstacles: {
    type: string;
    method: string;
    footprint_width_px: number;
    footprint_height_px: number;
    footprint_area_px: number;
    reduction_pct: number;
  }[];
}
