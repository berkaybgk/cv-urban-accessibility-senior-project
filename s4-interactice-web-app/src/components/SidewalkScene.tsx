"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { useState } from "react";
import type { ObstacleBox } from "@/lib/types";

interface SidewalkSceneProps {
  boxes: ObstacleBox[];
  stripHeight: number;
  stripWidth: number;
  pxToMeter?: number;
  side: "left" | "right";
  avgSidewalkWidthM?: number;
  targetSidewalkWidthPx?: number;
}

interface ObstacleProps {
  box: ObstacleBox;
  stripWidth: number;
  stripHeight: number;
  pxToMeter: number;
}

// Predefined physical dimensions (width, depth, height) in meters for each obstacle class
const PREDEFINED_DIMENSIONS: Record<
  string,
  { width: number; depth: number; height: number; color?: string }
> = {
  bollard: { width: 0.3, depth: 0.3, height: 0.9, color: "#737373" },
  car: { width: 1.8, depth: 4.5, height: 1.4, color: "#475569" },
  lamp_post: { width: 0.4, depth: 0.4, height: 6.0, color: "#94a3b8" },
  pedestrian: { width: 0.6, depth: 0.5, height: 1.7, color: "#f43f5e" },
  pedestrian_crossing: { width: 3.0, depth: 3.0, height: 0.05, color: "#ffffff" },
  road: { width: 6.0, depth: 6.0, height: 0.05, color: "#1e293b" },
  sidewalk: { width: 2.0, depth: 2.0, height: 0.1, color: "#c5d6e8" },
  traffic_light: { width: 0.3, depth: 0.3, height: 3.0, color: "#1e293b" },
  traffic_sign: { width: 0.5, depth: 0.1, height: 2.2, color: "#0284c7" },
  street_sign: { width: 0.5, depth: 0.1, height: 2.2, color: "#0284c7" }, // alias
  trash_container: { width: 0.8, depth: 0.8, height: 1.2, color: "#3b6e4c" },
  tree: { width: 2.0, depth: 2.0, height: 4.0, color: "#226622" },
  bench: { width: 1.5, depth: 0.6, height: 0.7, color: "#8c5630" },
};

// 3D representation of each obstacle
function ObstacleMesh({ box, stripWidth, stripHeight, pxToMeter }: ObstacleProps) {
  const [hovered, setHovered] = useState(false);

  const minRow = box.bbox[0];
  const minCol = box.bbox[1];
  const maxRow = box.bbox[2];
  const maxCol = box.bbox[3];

  const colCenter = (minCol + maxCol) / 2;
  const rowCenter = (minRow + maxRow) / 2;

  // Center the coordinate system so that the center of the strip is at x=0, z=0
  const x = (colCenter - stripWidth / 2) * pxToMeter;
  const z = (rowCenter - stripHeight / 2) * pxToMeter;

  // Resolve predefined dimensions
  const cleanClass = box.class_name.toLowerCase().replace(/[\s-]/g, "_");
  const predefined = PREDEFINED_DIMENSIONS[cleanClass];

  const widthM = predefined ? predefined.width : (maxCol - minCol) * pxToMeter;
  const depthM = predefined ? predefined.depth : (maxRow - minRow) * pxToMeter;
  const height = predefined ? predefined.height : 1.0;
  let color = predefined?.color || "#ff6b6b"; // Default color

  let geometry: React.ReactNode = null;

  switch (box.class_name) {
    case "bollard":
      color = hovered ? "#a3a3a3" : "#737373";
      geometry = (
        <meshStandardMaterial
          color={color}
          roughness={0.3}
          metalness={0.7}
        />
      );
      break;

    case "trash_container":
      color = hovered ? "#5c9c73" : "#3b6e4c";
      geometry = (
        <meshStandardMaterial
          color={color}
          roughness={0.7}
        />
      );
      break;

    case "tree":
      // Procedural tree height & geometry handles separately
      break;

    case "street_sign":
    case "traffic_sign":
      // Procedural signpost details
      break;

    case "traffic_light":
      // Procedural traffic light details
      break;

    case "bench":
      color = hovered ? "#c27d4c" : "#8c5630";
      break;

    default:
      // unknown obstacles
      color = hovered ? "#ff8c8c" : "#ef4444";
      geometry = (
        <meshStandardMaterial
          color={color}
          roughness={0.6}
        />
      );
      break;
  }

  // Position the model so that its bottom edge sits perfectly on the ground plane (y = 0)
  const y = height / 2;

  return (
    <group
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
      }}
      onPointerOut={() => setHovered(false)}
    >
      {box.class_name === "tree" ? (
        // Render tree procedurally (trunk cylinder + canopy sphere)
        <group position={[x, 0, z]}>
          {/* Trunk */}
          <mesh position={[0, height * 0.22, 0]} castShadow receiveShadow>
            <cylinderGeometry args={[widthM * 0.08, widthM * 0.11, height * 0.44, 8]} />
            <meshStandardMaterial color="#5c4033" roughness={0.9} />
          </mesh>
          {/* Canopy */}
          <mesh position={[0, height * 0.7, 0]} castShadow>
            <sphereGeometry args={[widthM * 0.45, 12, 12]} />
            <meshStandardMaterial color={hovered ? "#3c8a3c" : "#226622"} roughness={0.8} />
          </mesh>
        </group>
      ) : (box.class_name === "street_sign" || box.class_name === "traffic_sign") ? (
        // Render signpost procedurally
        <group position={[x, 0, z]}>
          {/* Metal Pole */}
          <mesh position={[0, height * 0.5, 0]} castShadow>
            <cylinderGeometry args={[0.03, 0.03, height, 8]} />
            <meshStandardMaterial color="#94a3b8" roughness={0.2} metalness={0.8} />
          </mesh>
          {/* Signboard */}
          <mesh position={[0, height * 0.9, 0]} castShadow>
            <boxGeometry args={[widthM, depthM * 3.5, 0.03]} />
            <meshStandardMaterial color={hovered ? "#38bdf8" : "#0284c7"} roughness={0.4} />
          </mesh>
        </group>
      ) : box.class_name === "traffic_light" ? (
        // Render traffic light procedurally
        <group position={[x, 0, z]}>
          {/* Dark Pole */}
          <mesh position={[0, height * 0.5, 0]} castShadow>
            <cylinderGeometry args={[0.04, 0.04, height, 8]} />
            <meshStandardMaterial color="#334155" roughness={0.5} />
          </mesh>
          {/* Light Box */}
          <mesh position={[0, height * 0.88, 0.06]} castShadow>
            <boxGeometry args={[widthM * 0.8, height * 0.23, depthM * 0.8]} />
            <meshStandardMaterial color="#1e293b" roughness={0.7} />
          </mesh>
          {/* Colored lights (Red, Yellow, Green bulbs) */}
          <mesh position={[0, height * 0.94, 0.185]}>
            <sphereGeometry args={[0.05, 8, 8]} />
            <meshBasicMaterial color="#ef4444" />
          </mesh>
          <mesh position={[0, height * 0.88, 0.185]}>
            <sphereGeometry args={[0.05, 8, 8]} />
            <meshBasicMaterial color="#eab308" />
          </mesh>
          <mesh position={[0, height * 0.82, 0.185]}>
            <sphereGeometry args={[0.05, 8, 8]} />
            <meshBasicMaterial color="#22c55e" />
          </mesh>
        </group>
      ) : box.class_name === "bench" ? (
        // Render detailed bench (seat + backrest)
        <group position={[x, 0, z]}>
          {/* Seat Plane */}
          <mesh position={[0, height * 0.35, 0]} castShadow receiveShadow>
            <boxGeometry args={[widthM, height * 0.08, depthM]} />
            <meshStandardMaterial color={color} roughness={0.8} />
          </mesh>
          {/* Backrest Plane */}
          <mesh position={[0, height * 0.78, -depthM * 0.44]} rotation={[0.1, 0, 0]} castShadow>
            <boxGeometry args={[widthM, height * 0.57, depthM * 0.1]} />
            <meshStandardMaterial color={color} roughness={0.8} />
          </mesh>
          {/* Leg Support Left */}
          <mesh position={[-widthM * 0.4, height * 0.21, 0]}>
            <boxGeometry args={[0.08, height * 0.42, depthM * 0.92]} />
            <meshStandardMaterial color="#475569" roughness={0.4} metalness={0.6} />
          </mesh>
          {/* Leg Support Right */}
          <mesh position={[widthM * 0.4, height * 0.21, 0]}>
            <boxGeometry args={[0.08, height * 0.42, depthM * 0.92]} />
            <meshStandardMaterial color="#475569" roughness={0.4} metalness={0.6} />
          </mesh>
        </group>
      ) : box.class_name === "bollard" ? (
        // Cylindrical bollard mesh
        <mesh position={[x, y, z]} castShadow receiveShadow>
          <cylinderGeometry args={[widthM / 2, widthM / 2, height, 16]} />
          {geometry}
        </mesh>
      ) : (
        // Box fallback for cars / pedestrians / road / sidewalk / trash containers / unknown boxes
        <mesh position={[x, y, z]} castShadow receiveShadow>
          <boxGeometry args={[widthM, height, depthM]} />
          {geometry}
        </mesh>
      )}

      {/* HTML Hover Tooltip */}
      {hovered && (
        <Html position={[x, height + 0.3, z]} center distanceFactor={8}>
          <div className="pointer-events-none select-none rounded bg-neutral-900/95 px-2.5 py-1.5 text-[10px] text-neutral-200 border border-neutral-700/80 shadow-lg backdrop-blur-sm whitespace-nowrap z-50">
            <div className="font-semibold text-cyan-400 capitalize">
              {box.class_name.replace("_", " ")}
            </div>
            <div className="text-[9px] text-neutral-400 mt-0.5">
              Size: {widthM.toFixed(2)}m × {depthM.toFixed(2)}m × {height.toFixed(1)}m
            </div>
            {box.tile_label && (
              <div className="text-[9px] text-neutral-400 mt-0.5">
                Src: {box.tile_label}
              </div>
            )}
          </div>
        </Html>
      )}
    </group>
  );
}

export default function SidewalkScene({
  boxes,
  stripHeight,
  stripWidth,
  pxToMeter: propPxToMeter = 0.05,
  side,
  avgSidewalkWidthM,
  targetSidewalkWidthPx,
}: SidewalkSceneProps) {
  // Compute pxToMeter dynamically based on estimated physical width
  const pxToMeter = avgSidewalkWidthM && targetSidewalkWidthPx
    ? avgSidewalkWidthM / targetSidewalkWidthPx
    : propPxToMeter;

  // Convert full dimensions to meters
  const actualSidewalkWidthM = avgSidewalkWidthM || (stripWidth * pxToMeter);
  const sidewalkLengthM = stripHeight * pxToMeter;
  const roadWidthM = 5.0;

  // Road placement relative to the sidewalk
  // If we are looking at the RIGHT sidewalk, the road is on the left (-x direction)
  // If we are looking at the LEFT sidewalk, the road is on the right (+x direction)
  const roadX = side === "right"
    ? -(actualSidewalkWidthM / 2 + roadWidthM / 2)
    : (actualSidewalkWidthM / 2 + roadWidthM / 2);

  // Position camera slightly offset from center to show the segment
  const cameraZPosition = sidewalkLengthM / 2; // view near the center of the segment

  return (
    <Canvas
      shadows
      camera={{
        position: [actualSidewalkWidthM * 1.5, 4.0, cameraZPosition],
        fov: 50,
      }}
    >
      <color attach="background" args={["#0a0a0a"]} />

      {/* Lighting */}
      <ambientLight intensity={1.2} />
      <directionalLight
        position={[20, 35, 10]}
        intensity={1.5}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-bias={-0.0001}
      />

      {/* Sidewalk Ground Plane */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.005, 0]}
        receiveShadow
      >
        <planeGeometry args={[actualSidewalkWidthM, sidewalkLengthM]} />
        <meshStandardMaterial
          color="#c5d6e8" // sidewalk tint matching #bed7f5, a bit brighter under light
          roughness={0.7}
        />
      </mesh>

      {/* Curb highlight line */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[
          side === "right" ? -actualSidewalkWidthM / 2 : actualSidewalkWidthM / 2,
          0.008,
          0,
        ]}
      >
        <planeGeometry args={[0.15, sidewalkLengthM]} />
        <meshStandardMaterial color="#cbd5e1" roughness={0.5} />
      </mesh>

      {/* Asphalt Road Ground Plane */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[roadX, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[roadWidthM, sidewalkLengthM]} />
        <meshStandardMaterial
          color="#1e293b" // slate-800 asphalt road
          roughness={0.9}
        />
      </mesh>

      {/* Road Lane Markings */}
      {/* Dashed white/yellow road line in the middle of the road */}
      <group position={[roadX, 0.002, 0]}>
        {/* Draw markings every 4 meters along the length of the road */}
        {Array.from({ length: Math.ceil(sidewalkLengthM / 4) }).map((_, i) => {
          const zOffset = i * 4 - sidewalkLengthM / 2 + 2;
          if (zOffset > sidewalkLengthM / 2) return null;
          return (
            <mesh
              key={i}
              rotation={[-Math.PI / 2, 0, 0]}
              position={[0, 0, zOffset]}
            >
              <planeGeometry args={[0.1, 1.8]} />
              <meshBasicMaterial color="#eab308" /> {/* yellow lane marker */}
            </mesh>
          );
        })}
      </group>

      {/* Render each footprint box */}
      {boxes.map((box, idx) => (
        <ObstacleMesh
          key={`${box.class_name}-${idx}`}
          box={box}
          stripWidth={stripWidth}
          stripHeight={stripHeight}
          pxToMeter={pxToMeter}
        />
      ))}

      {/* Controls */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        maxPolarAngle={Math.PI / 2 - 0.02} // don't go below ground level
        minDistance={1}
        maxDistance={sidewalkLengthM * 1.5}
      />
    </Canvas>
  );
}

