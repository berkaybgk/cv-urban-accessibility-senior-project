import { NextRequest, NextResponse } from "next/server";

const STRIP_SERVICE_URL = process.env.STRIP_SERVICE_URL || "http://localhost:8000";

// Computing previews downloads imagery + fits edges per tile; give it room.
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const res = await fetch(`${STRIP_SERVICE_URL}/strip/edges`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const text = await res.text();
    if (!res.ok) {
      console.error(`[strip/edges] service ${res.status}: ${text.slice(0, 500)}`);
      return NextResponse.json(
        { error: "Strip service failed to compute edges", detail: text.slice(0, 500) },
        { status: res.status }
      );
    }
    return new NextResponse(text, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("[strip/edges] proxy error:", err);
    return NextResponse.json(
      { error: `Could not reach strip service at ${STRIP_SERVICE_URL}` },
      { status: 502 }
    );
  }
}
