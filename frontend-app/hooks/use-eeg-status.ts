"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { EegStatus } from "@/lib/neurim-types";

const POLL_INTERVAL_MS = 2000;

export interface EegStatusHook {
  status: EegStatus | null;
  // false when the Next proxy / api_server.py cannot be reached (backend down).
  reachable: boolean;
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
      } else {
        // Proxy answered but api_server.py is unreachable (502) or malformed.
        setReachable(false);
      }
    } catch {
      if (mounted.current) setReachable(false);
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
      if (!mounted.current) return;
      setReachable(res.ok || res.status !== 502);
    } catch {
      if (mounted.current) setReachable(false);
    } finally {
      if (mounted.current) setRetrying(false);
      // Truth resumes from the regular poll; don't adopt the retry response's
      // transient "disconnected" reset here.
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

  return { status, reachable, loading, retrying, retry };
}
