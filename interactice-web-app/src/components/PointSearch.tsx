"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PointData, PointsHashMap, Direction } from "@/lib/types";

interface PointSearchProps {
  points: PointsHashMap;
  onSelectPoint: (point: PointData) => void;
  onSelectDirection: (point: PointData, direction: Direction) => void;
}

export default function PointSearch({
  points,
  onSelectPoint,
  onSelectDirection,
}: PointSearchProps) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const allIds = useMemo(() => Object.keys(points).sort(), [points]);

  const results = useMemo(() => {
    if (!query.trim()) return allIds.slice(0, 8);
    return allIds.filter((id) => id.includes(query.trim())).slice(0, 20);
  }, [allIds, query]);

  const showDropdown = focused && results.length > 0;

  useEffect(() => {
    setSelectedIdx(0);
  }, [results]);

  const handleSelect = useCallback(
    (pointId: string) => {
      const pt = points[pointId];
      if (pt) {
        onSelectPoint(pt);
        setQuery(pointId);
        setFocused(false);
        inputRef.current?.blur();
      }
    },
    [points, onSelectPoint]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!showDropdown) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIdx((i) => Math.min(i + 1, results.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIdx((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (results[selectedIdx]) handleSelect(results[selectedIdx]);
      } else if (e.key === "Escape") {
        setFocused(false);
        inputRef.current?.blur();
      }
    },
    [showDropdown, results, selectedIdx, handleSelect]
  );

  useEffect(() => {
    if (showDropdown && listRef.current) {
      const active = listRef.current.children[selectedIdx] as HTMLElement;
      active?.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIdx, showDropdown]);

  return (
    <div className="relative">
      <div className="flex items-center gap-2 bg-neutral-900/80 backdrop-blur-sm
        rounded-lg border border-neutral-700/50 px-3 py-1.5">
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
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => {
            setTimeout(() => setFocused(false), 150);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Search point ID..."
          className="bg-transparent text-sm text-white placeholder-neutral-500
            outline-none w-36 sm:w-44"
        />
        {query && (
          <button
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => {
              setQuery("");
              inputRef.current?.focus();
            }}
            className="text-neutral-500 hover:text-neutral-300 text-xs leading-none"
          >
            &times;
          </button>
        )}
      </div>

      {showDropdown && (
        <div
          ref={listRef}
          className="absolute top-full left-0 mt-1 w-full max-h-64 overflow-y-auto
            bg-neutral-900/95 backdrop-blur-sm rounded-lg border border-neutral-700/50
            shadow-xl z-50"
        >
          {results.map((id, idx) => {
            const pt = points[id];
            return (
              <button
                key={id}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => handleSelect(id)}
                className={`w-full text-left px-3 py-2 text-xs transition-colors flex items-center justify-between
                  ${
                    idx === selectedIdx
                      ? "bg-neutral-700/60 text-white"
                      : "text-neutral-300 hover:bg-neutral-700/40 hover:text-white"
                  }`}
              >
                <span className="font-mono font-medium">{id}</span>
                <span className="text-neutral-500 text-[10px]">
                  {pt.latitude.toFixed(4)}, {pt.longitude.toFixed(4)}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
