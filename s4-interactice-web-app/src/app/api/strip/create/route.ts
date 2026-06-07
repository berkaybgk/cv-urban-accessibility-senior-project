import { NextRequest, NextResponse } from "next/server";

const STRIP_SERVICE_URL = process.env.STRIP_SERVICE_URL || "http://localhost:8000";

// Full strip build runs LoFTR over every tile pair; allow a long window.
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const res = await fetch(`${STRIP_SERVICE_URL}/strip/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const text = await res.text();
      console.error(`[strip/create] service ${res.status}: ${text.slice(0, 500)}`);
      return NextResponse.json(
        { error: "Strip service failed to build strip", detail: text.slice(0, 500) },
        { status: res.status }
      );
    }

    // Stream the zip back, forwarding the strip identifiers.
    const buf = await res.arrayBuffer();
    return new NextResponse(buf, {
      status: 200,
      headers: {
        "Content-Type": res.headers.get("Content-Type") || "application/zip",
        "Content-Disposition": res.headers.get("Content-Disposition") || 'attachment; filename="strip.zip"',
        "X-Strip-Id": res.headers.get("X-Strip-Id") || "",
        "X-Strip-Gcs-Prefix": res.headers.get("X-Strip-Gcs-Prefix") || "",
      },
    });
  } catch (err) {
    console.error("[strip/create] proxy error:", err);
    return NextResponse.json(
      { error: `Could not reach strip service at ${STRIP_SERVICE_URL}` },
      { status: 502 }
    );
  }
}
