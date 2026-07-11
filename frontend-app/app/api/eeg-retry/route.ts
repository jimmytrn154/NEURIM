import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const DEFAULT_API_URL = "http://127.0.0.1:8000";

function cleanUrl(value: string | undefined, fallback: string) {
  const cleaned = value?.trim().replace(/\/$/, "");
  return cleaned || fallback;
}

export async function POST() {
  const apiBase = cleanUrl(process.env.NEURIM_API_URL, DEFAULT_API_URL);
  try {
    const response = await fetch(`${apiBase}/eeg/retry`, {
      method: "POST",
      cache: "no-store",
    });
    const json = await response.json().catch(() => ({}));
    return NextResponse.json(json, { status: response.status });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: `Could not reach local api_server.py at ${apiBase}: ${message}`, backend_url: apiBase },
      { status: 502 }
    );
  }
}
