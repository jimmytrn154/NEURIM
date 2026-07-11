import { randomUUID } from "node:crypto";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

type SessionIntentRequest = {
  prompt?: unknown;
  mock?: unknown;
  baseline_seconds?: unknown;
  server_url?: unknown;
};

type BackendSession = {
  running: boolean;
  pid: number | null;
  started_at: string | null;
  prompt: string | null;
  exit_code: number | null;
  manifest_path?: string | null;
};

const MAX_STEPS = 100;
const DEFAULT_API_URL = "http://127.0.0.1:8000";
const DEFAULT_RENDER_SERVER_URL = "http://localhost:8766";

function cleanUrl(value: string | undefined, fallback: string) {
  const cleaned = value?.trim().replace(/\/$/, "");
  return cleaned || fallback;
}

function requestString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function requestBoolean(value: unknown, fallback: boolean) {
  return typeof value === "boolean" ? value : fallback;
}

function requestNumber(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function backendError(payload: Record<string, unknown>, fallback: string) {
  if (typeof payload.error === "string") return payload.error;
  if (typeof payload.detail === "string") return payload.detail;
  if (Array.isArray(payload.detail)) {
    return payload.detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item && typeof item.msg === "string") return item.msg;
        return JSON.stringify(item);
      })
      .join("; ");
  }
  return fallback;
}

export async function POST(request: Request) {
  let body: SessionIntentRequest;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON request body" }, { status: 400 });
  }

  const prompt = typeof body.prompt === "string" ? body.prompt.trim() : "";
  if (!prompt) {
    return NextResponse.json({ error: "prompt is required" }, { status: 400 });
  }

  const apiBase = cleanUrl(process.env.NEURIM_API_URL, DEFAULT_API_URL);
  const baselineFallback = Number(process.env.NEURIM_BASELINE_SECONDS ?? 0);
  const startPayload = {
    prompt,
    mock: requestBoolean(body.mock, false),
    baseline_seconds: requestNumber(body.baseline_seconds, Number.isFinite(baselineFallback) ? baselineFallback : 0),
    server_url: requestString(body.server_url) || cleanUrl(process.env.NEURIM_DIFFUSION_SERVER_URL, DEFAULT_RENDER_SERVER_URL),
  };

  try {
    const response = await fetch(`${apiBase}/session/start`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(startPayload),
      cache: "no-store",
    });
    const json = (await response.json().catch(() => ({}))) as Record<string, unknown>;
    if (!response.ok) {
      return NextResponse.json(
        {
          error: backendError(json, `api_server.py returned HTTP ${response.status}`),
          backend_url: apiBase,
          backend_session: json,
        },
        { status: response.status }
      );
    }

    const backendSession = json as BackendSession;
    return NextResponse.json({
      ok: true,
      session_id: backendSession.pid ? `local-${backendSession.pid}` : randomUUID(),
      prompt: backendSession.prompt || prompt,
      max_steps: MAX_STEPS,
      prompt_routing: "local_api_server",
      render_contract: "z_to_render_server",
      backend_url: apiBase,
      backend_session: backendSession,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error: `Could not reach local api_server.py at ${apiBase}: ${message}`,
        backend_url: apiBase,
      },
      { status: 502 }
    );
  }
}
