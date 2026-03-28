import { NextResponse } from "next/server";
import { getPoints } from "@/lib/csvParser";

export async function GET() {
  try {
    const points = getPoints();
    return NextResponse.json(points);
  } catch (err) {
    console.error("Failed to parse points CSV:", err);
    return NextResponse.json(
      { error: "Failed to load points data" },
      { status: 500 }
    );
  }
}
