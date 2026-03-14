import { NextRequest, NextResponse } from "next/server";
import { uploadJSON } from "@/lib/gcs";

const REPORTS_PREFIX = "reports";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { pointId, direction, latitude, longitude, description } = body;

    if (!pointId || !direction || !description?.trim()) {
      return NextResponse.json(
        { error: "Missing required fields: pointId, direction, description" },
        { status: 400 }
      );
    }

    const timestamp = new Date().toISOString();
    const safeTimestamp = timestamp.replace(/[:.]/g, "-");
    const blobPath = `${REPORTS_PREFIX}/${pointId}/${direction}/${safeTimestamp}.json`;

    const report = {
      pointId,
      direction,
      latitude,
      longitude,
      description: description.trim(),
      timestamp,
    };

    const uri = await uploadJSON(blobPath, report);

    return NextResponse.json({ success: true, uri, report });
  } catch (err) {
    console.error("Failed to save report:", err);
    return NextResponse.json(
      { error: "Failed to save report" },
      { status: 500 }
    );
  }
}
