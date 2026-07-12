"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { EegStatus } from "@/lib/neurim-types";

const POLL_INTERVAL_MS = 2000;

export interface EegStatusHook {
  status: EegStatus | null;
  // false when the Next proxy / api_server.py cannot be reached (backend down).
  reachable: boolean;
  // proxy/backend error text when no valid EEG status payload is available.
  error: string | null;
  // true until the first status check resolves.
  loading: boolean;
  // true while a manual retry request is in flight.
  retrying: boolean;
  retry: () => Promise<void>;
}

function isEegStatus(value: unknown): value is EegStatus {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { state?: unknown }).state === "string"
  );
}

/**
 * Polls the EEG connection lifecycle from `/api/eeg-status` (which proxies
 * `api_server.py /eeg/status`). `reachable` distinguishes "backend down" from
 * "headset disconnected" so the caller can keep the offline/mock experience
 * working when there is no local api_server at all. `retry` kicks the backend
 * connector and optimistically shows the connecting state so the UI reacts
 * immediately rather than waiting for the next poll.
 */
export function useEegStatus(): EegStatusHook {
  const [status, setStatus] = useState<EegStatus | null>(null);
  const [reachable, setReachable] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retrying, setRetrying] = useState(false);
  const mounted = useRef(true);

  const poll = useCallback(async () => {
    try {
      const res = await fetch("/api/eeg-status", { cache: "no-store" });
      const json = await res.json().catch(() => null);
      if (!mounted.current) return;
      if (res.ok && isEegStatus(json)) {
        setStatus(json);
        setReachable(true);
        setError(null);
      } else {
        // Proxy answered but api_server.py is unreachable (502) or malformed.
        const message =
          typeof (json as { error?: unknown } | null)?.error === "string"
            ? String((json as { error: string }).error)
            : `EEG status request failed with HTTP ${res.status}`;
        setReachable(false);
        setError(message);
      }
    } catch (pollError) {
      if (mounted.current) {
        const message = pollError instanceof Error ? pollError.message : String(pollError);
        setReachable(false);
        setError(message);
      }
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  const retry = useCallback(async () => {
    setRetrying(true);
    // Optimistically flip to connecting so the animation shows without waiting
    // for the connector thread to advance past the reset "disconnected" state.
    setStatus((prev) =>
      prev ? { ...prev, state: "connecting", last_error: null, next_retry_at: null } : prev
    );
    try {
      const res = await fetch("/api/eeg-retry", { method: "POST", cache: "no-store" });
      const json = await res.json().catch(() => null);
      if (!mounted.current) return;
      if (res.ok && isEegStatus(json)) {
        setStatus(json);
        setReachable(true);
        setError(null);
      } else {
        const message =
          typeof (json as { error?: unknown } | null)?.error === "string"
            ? String((json as { error: string }).error)
            : `EEG retry request failed with HTTP ${res.status}`;
        setReachable(false);
        setError(message);
      }
    } catch (retryError) {
      if (mounted.current) {
        const message = retryError instanceof Error ? retryError.message : String(retryError);
        setReachable(false);
        setError(message);
      }
    } finally {
      if (mounted.current) setRetrying(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    // Deferred so the first poll runs as a timer callback rather than a
    // synchronous setState inside the effect body.
    const initial = setTimeout(poll, 0);
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      mounted.current = false;
      clearTimeout(initial);
      clearInterval(id);
    };
  }, [poll]);

  return { status, reachable, error, loading, retrying, retry };
}
