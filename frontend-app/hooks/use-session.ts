"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePngFrame } from "@/hooks/use-png-frame";
import { makeMockSession } from "@/lib/mock-frame";
import type { BackendSession, FrameMessage, SessionIntentResponse } from "@/lib/neurim-types";

export type SessionPhase = "idle" | "processing" | "live" | "finalizing" | "completed";

export interface NeurimSession {
  phase: SessionPhase;
  submittedPrompt: string;
  sessionId: string | null;
  statusText: string;
  offline: boolean;
  isSubmitting: boolean;
  connected: boolean;
  frame: FrameMessage | null;
  frameSrc: string | null;
  fps: number;
  reward: number;
  finalSrc: string | null;
  completed: boolean;
  resultRefined: boolean;
  finalizeError: string | null;
  isRetryingFinalization: boolean;
  showBrain: boolean;
  setShowBrain: (value: boolean) => void;
  startSession: (prompt: string) => Promise<void>;
  retryFinalization: () => Promise<void>;
  reset: () => void;
}

class SessionStartError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "SessionStartError";
    this.status = status;
  }
}

export function useSession(): NeurimSession {
  const [active, setActive] = useState(false);
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("Ready");
  const [offline, setOffline] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showBrain, setShowBrain] = useState(true);
  const [offlineFrame, setOfflineFrame] = useState<FrameMessage | null>(null);
  const [offlineSrc, setOfflineSrc] = useState<string | null>(null);
  const [backendPhase, setBackendPhase] = useState<BackendSession["phase"]>("idle");
  const [finalSrc, setFinalSrc] = useState<string | null>(null);
  const [resultRefined, setResultRefined] = useState(false);
  const [finalizeError, setFinalizeError] = useState<string | null>(null);
  const [isRetryingFinalization, setIsRetryingFinalization] = useState(false);
  const finalUrlRef = useRef<string | null>(null);
  const { src: liveSrc, available: liveAvailable, reset: resetLive } = usePngFrame(
    active && !offline && backendPhase !== "completed",
  );

  const revokeFinal = useCallback(() => {
    if (finalUrlRef.current) URL.revokeObjectURL(finalUrlRef.current);
    finalUrlRef.current = null;
  }, []);

  const loadFinalFrame = useCallback(async () => {
    const response = await fetch(`/api/target-frame?ts=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) return false;
    const nextUrl = URL.createObjectURL(await response.blob());
    revokeFinal();
    finalUrlRef.current = nextUrl;
    setFinalSrc(nextUrl);
    return true;
  }, [revokeFinal]);

  useEffect(() => () => revokeFinal(), [revokeFinal]);

  useEffect(() => {
    if (!active || offline || !sessionId) return;
    let cancelled = false;
    let inFlight = false;

    const poll = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const response = await fetch("/api/session-status", { cache: "no-store" });
        if (!response.ok || cancelled) return;
        const status = (await response.json()) as BackendSession;
        setBackendPhase(status.phase);
        setResultRefined(status.result_refined);
        setFinalizeError(status.finalize_error);

        if (status.phase === "finalizing") {
          setStatusText("Refining the final image with OpenAI");
        } else if (status.phase === "failed") {
          setStatusText(status.finalize_error || "Session failed");
        } else if (status.phase === "completed" && status.result_ready && !finalSrc) {
          if (await loadFinalFrame()) {
            setStatusText(status.result_refined ? "Final image ready" : "Unrefined final frame ready");
            setIsRetryingFinalization(false);
          }
        }
      } catch {
        // Preserve the current view and retry while the local API restarts.
      } finally {
        inFlight = false;
      }
    };

    poll();
    const interval = setInterval(poll, 500);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [active, offline, sessionId, finalSrc, loadFinalFrame]);

  const startSession = useCallback(async (rawPrompt: string) => {
    const prompt = rawPrompt.trim();
    if (!prompt) {
      setStatusText("Write a prompt first.");
      return;
    }

    setIsSubmitting(true);
    setActive(true);
    setOffline(false);
    setBackendPhase("running");
    setFinalSrc(null);
    setResultRefined(false);
    setFinalizeError(null);
    revokeFinal();
    resetLive();
    setStatusText("Rendering the first frame");

    try {
      const response = await fetch("/api/session-intent", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const json = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new SessionStartError(json.error || "Failed to start session", response.status);
      }
      const intent = json as SessionIntentResponse;
      setSubmittedPrompt(intent.prompt || prompt);
      setSessionId(intent.session_id);
      setBackendPhase(intent.backend_session.phase || "running");
      setStatusText("Session running");
    } catch (error) {
      if (error instanceof SessionStartError && error.status !== 502) {
        setActive(false);
        setOffline(false);
        setSessionId(null);
        setBackendPhase("idle");
        setStatusText(error.message);
        return;
      }
      const fallback = makeMockSession(prompt, Date.now() / 1000);
      setSubmittedPrompt(prompt);
      setSessionId(`local-${Date.now().toString(36).slice(-6)}`);
      setOffline(true);
      setOfflineFrame(fallback.frame);
      setOfflineSrc(fallback.candidateSrc);
      setStatusText(error instanceof Error ? error.message : "Offline preview");
    } finally {
      setIsSubmitting(false);
    }
  }, [resetLive, revokeFinal]);

  const retryFinalization = useCallback(async () => {
    setIsRetryingFinalization(true);
    const response = await fetch("/api/finalize-retry", { method: "POST", cache: "no-store" });
    const json = await response.json().catch(() => ({}));
    if (!response.ok) {
      setIsRetryingFinalization(false);
      setFinalizeError(json.detail || json.error || "Could not retry refinement");
      return;
    }
    setFinalSrc(null);
    revokeFinal();
    setBackendPhase("finalizing");
    setFinalizeError(null);
    setStatusText("Retrying OpenAI refinement");
  }, [revokeFinal]);

  const reset = useCallback(() => {
    resetLive();
    revokeFinal();
    setActive(false);
    setSubmittedPrompt("");
    setSessionId(null);
    setStatusText("Ready");
    setOffline(false);
    setOfflineFrame(null);
    setOfflineSrc(null);
    setBackendPhase("idle");
    setFinalSrc(null);
    setResultRefined(false);
    setFinalizeError(null);
    setIsRetryingFinalization(false);
  }, [resetLive, revokeFinal]);

  const completed = backendPhase === "completed" && Boolean(finalSrc);
  const phase: SessionPhase = !active
    ? "idle"
    : completed
      ? "completed"
      : backendPhase === "finalizing"
        ? "finalizing"
        : (liveAvailable || offlineSrc)
          ? "live"
          : "processing";
  const frame = offline ? offlineFrame : null;
  const frameSrc = offline ? offlineSrc : liveSrc;

  return {
    phase,
    submittedPrompt,
    sessionId,
    statusText,
    offline,
    isSubmitting,
    connected: liveAvailable,
    frame,
    frameSrc,
    fps: 0,
    reward: frame?.eeg_features?.faa.reward ?? frame?.reward_estimate ?? 0,
    finalSrc,
    completed,
    resultRefined,
    finalizeError,
    isRetryingFinalization,
    showBrain,
    setShowBrain,
    startSession,
    retryFinalization,
    reset,
  };
}
