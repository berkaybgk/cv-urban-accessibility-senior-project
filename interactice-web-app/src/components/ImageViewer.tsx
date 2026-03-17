"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface ImageViewerProps {
  src: string;
  alt: string;
  className?: string;
}

export default function ImageViewer({
  src,
  alt,
  className = "",
}: ImageViewerProps) {
  const [loaded, setLoaded] = useState(false);
  const [errored, setErrored] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isPanning = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const handleOpen = useCallback(() => {
    resetView();
    setExpanded(true);
  }, [resetView]);

  const handleClose = useCallback(() => {
    setExpanded(false);
    resetView();
  }, [resetView]);

  useEffect(() => {
    if (!expanded) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleClose();
      if (e.key === "+" || e.key === "=")
        setZoom((z) => Math.min(z * 1.3, 10));
      if (e.key === "-") setZoom((z) => Math.max(z / 1.3, 0.5));
      if (e.key === "0") resetView();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [expanded, handleClose, resetView]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setZoom((z) => Math.max(0.5, Math.min(10, z * factor)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    isPanning.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    setPan((p) => ({ x: p.x + dx, y: p.y + dy }));
  }, []);

  const handleMouseUp = useCallback(() => {
    isPanning.current = false;
  }, []);

  if (errored) {
    return (
      <div
        className={`flex items-center justify-center bg-neutral-800 rounded-lg text-neutral-500 text-xs ${className}`}
      >
        Failed to load image
      </div>
    );
  }

  return (
    <>
      <div
        className={`relative overflow-hidden rounded-lg bg-neutral-800 ${className}`}
      >
        {!loaded && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-neutral-400 border-t-transparent" />
          </div>
        )}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={alt}
          className={`w-full h-full object-contain cursor-zoom-in transition-opacity ${
            loaded ? "opacity-100" : "opacity-0"
          }`}
          onLoad={() => setLoaded(true)}
          onError={() => setErrored(true)}
          onClick={handleOpen}
        />
      </div>

      {expanded && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/95"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ cursor: zoom > 1 ? "grab" : "zoom-in" }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={alt}
            className="max-w-none select-none pointer-events-none"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              maxWidth: zoom <= 1 ? "95vw" : "none",
              maxHeight: zoom <= 1 ? "95vh" : "none",
              objectFit: "contain",
            }}
            draggable={false}
          />

          {/* Controls bar */}
          <div
            className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2
            bg-neutral-900/80 backdrop-blur-sm rounded-full px-4 py-2 border border-neutral-700/50"
          >
            <button
              onClick={(e) => {
                e.stopPropagation();
                setZoom((z) => Math.max(z / 1.3, 0.5));
              }}
              className="text-neutral-300 hover:text-white px-2 py-0.5 text-sm font-medium transition-colors"
              title="Zoom out (-)"
            >
              &minus;
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                resetView();
              }}
              className="text-neutral-400 hover:text-white px-2 py-0.5 text-xs font-medium transition-colors"
              title="Reset zoom (0)"
            >
              {Math.round(zoom * 100)}%
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setZoom((z) => Math.min(z * 1.3, 10));
              }}
              className="text-neutral-300 hover:text-white px-2 py-0.5 text-sm font-medium transition-colors"
              title="Zoom in (+)"
            >
              +
            </button>
          </div>

          <button
            onClick={handleClose}
            className="absolute top-4 right-4 text-white/70 hover:text-white
              text-2xl leading-none p-2 rounded-full hover:bg-white/10 transition-colors"
          >
            &times;
          </button>

          <div className="absolute top-4 left-4 text-neutral-500 text-xs">
            Scroll to zoom &middot; Drag to pan &middot; Press 0 to reset
          </div>
        </div>
      )}
    </>
  );
}
