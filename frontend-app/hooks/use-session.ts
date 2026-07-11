"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useFrameStream } from "@/hooks/use-frame-stream";
import { makeMockSession, type MockSession } from "@/lib/mock-frame";
import type { FrameMessage, SessionIntentResponse } from "@/lib/neurim-types";

export type SessionPhase = "idle" | "processing" | "live";

// The processing beat before the mock frame is revealed when no real frame
// has arrived from an auto-connected hub.
const REVEAL_DELAY_MS = 1400;

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

  wsUrl: string;
  setWsUrl: (url: string) => void;

  showBrain: boolean;
  setShowBrain: (value: boolean) => void;

  startSession: (prompt: string) => Promise<void>;
  reset: () => void;
}

/**
 * Top-level session orchestration. Posts the prompt intent (falling back to an
 * offline/mock session when the backend is down), runs the processing beat, and
 * merges the live frame stream with the deterministic mock so the display
 * always has something to show.
 */
export function useSession(): NeurimSession {
  const [active, setActive] = useState(false);
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("Ready");
  const [offline, setOffline] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showBrain, setShowBrain] = useState(true);
  const [mock, setMock] = useState<MockSession | null>(null);
  // Set true only when the fallback timer fires; a real frame reveals via render.
  const [timedOut, setTimedOut] = useState(false);
  const revealTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stream = useFrameStream(active);

  useEffect(() => {
    return () => {
      if (revealTimer.current) clearTimeout(revealTimer.current);
    };
  }, []);

  const startSession = useCallback(async (rawPrompt: string) => {
    const prompt = rawPrompt.trim();
    if (!prompt) {
      setStatusText("Write a prompt first.");
      return;
    }

    setIsSubmitting(true);
    setActive(true);
    setTimedOut(false);
    setStatusText("Reading your signal — rendering the first frame");

    // Fall back to the mock frame if auto-connect delivers nothing in time.
    if (revealTimer.current) clearTimeout(revealTimer.current);
    revealTimer.current = setTimeout(() => setTimedOut(true), REVEAL_DELAY_MS);

    try {
      const response = await fetch("/api/session-intent", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const json = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(json.error || "Failed to save session intent");
      const intent = json as SessionIntentResponse;
      const accepted = intent.prompt || prompt;
      setSubmittedPrompt(accepted);
      setSessionId(intent.session_id);
      setOffline(false);
      setMock(makeMockSession(accepted, Date.now() / 1000));
      setStatusText(
        intent.backend_session.pid
          ? `api_server.py accepted prompt · pid ${intent.backend_session.pid}`
          : "api_server.py accepted prompt"
      );
    } catch {
      // Resilience: still enter the session view in an offline/mock state.
      setSubmittedPrompt(prompt);
      setSessionId(`local-${Date.now().toString(36).slice(-6)}`);
      setOffline(true);
      setMock(makeMockSession(prompt, Date.now() / 1000));
      setStatusText("Offline · showing mock preview");
    } finally {
      setIsSubmitting(false);
    }
  }, []);

  const reset = useCallback(() => {
    if (revealTimer.current) clearTimeout(revealTimer.current);
    stream.disconnect();
    setActive(false);
    setTimedOut(false);
    setSubmittedPrompt("");
    setSessionId(null);
    setStatusText("Ready");
    setOffline(false);
    setMock(null);
  }, [stream]);

  // Derive the phase: a real frame OR the fallback timer ends the processing beat.
  const revealed = timedOut || stream.frame != null;
  const phase: SessionPhase = !active ? "idle" : revealed ? "live" : "processing";

  // Prefer the live frame; fall back to the revealed mock.
  const frame = stream.frame ?? (revealed ? mock?.frame ?? null : null);
  const frameSrc = stream.frameSrc ?? (revealed ? mock?.candidateSrc ?? null : null);
  const fps = stream.connected && stream.frame ? stream.fps : 24;
  const reward = frame?.eeg_features?.faa.reward ?? frame?.reward_estimate ?? 0;

  return {
    phase,
    submittedPrompt,
    sessionId,
    statusText,
    offline,
    isSubmitting,
    connected: stream.connected,
    frame,
    frameSrc,
    fps,
    reward,
    wsUrl: stream.url,
    setWsUrl: stream.setUrl,
    showBrain,
    setShowBrain,
    startSession,
    reset,
  };
}
