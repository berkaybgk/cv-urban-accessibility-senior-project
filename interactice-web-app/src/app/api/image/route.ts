import { NextRequest, NextResponse } from "next/server";
import { downloadBlob, validateBlobPath } from "@/lib/gcs";

const CONTENT_TYPES: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".json": "application/json",
};

function getContentType(blobPath: string): string {
  const ext = blobPath.slice(blobPath.lastIndexOf(".")).toLowerCase();
  return CONTENT_TYPES[ext] || "application/octet-stream";
}

export async function GET(request: NextRequest) {
  const blob = request.nextUrl.searchParams.get("blob");

  if (!blob) {
    return NextResponse.json(
      { error: "Missing 'blob' query parameter" },
      { status: 400 }
    );
  }

  if (!validateBlobPath(blob)) {
    return NextResponse.json(
      { error: "Blob path not allowed" },
      { status: 403 }
    );
  }

  try {
    const data = await downloadBlob(blob);
    const contentType = getContentType(blob);

    return new NextResponse(new Uint8Array(data), {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=86400, immutable",
      },
    });
  } catch (err) {
    console.error(`Failed to download blob ${blob}:`, err);
    return NextResponse.json(
      { error: "Failed to fetch image from storage" },
      { status: 404 }
    );
  }
}
