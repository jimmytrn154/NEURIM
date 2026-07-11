"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { decodeFrameSrc, type FrameMessage } from "@/lib/neurim-types";

export const DEFAULT_HUB_URL = "ws://localhost:8765";

export interface FrameStream {
  url: string;
  setUrl: (url: string) => void;
  connected: boolean;
  status: string;
  frame: FrameMessage | null;
  frameSrc: string | null;
  fps: number;
  connect: () => void;
  disconnect: () => void;
}

/**
 * Owns the display-side WebSocket to the frame hub. Auto-connects whenever
 * `enabled` is true (session start), sends the `{role:"display"}` handshake,
 * parses frames, and tracks fps. Editing `url` while enabled reconnects.
 * Offline simply yields `connected=false` with a null frame — the caller
 * layers the mock fallback on top.
 */
export function useFrameStream(enabled: boolean): FrameStream {
  const [url, setUrl] = useState(DEFAULT_HUB_URL);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("Not connected");
  const [frame, setFrame] = useState<FrameMessage | null>(null);
  const [frameSrc, setFrameSrc] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const frameCounter = useRef({ count: 0, startedAt: 0 });

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
    setStatus("Disconnected");
  }, []);

  const connect = useCallback(() => {
    if (typeof window === "undefined") return;
    if (wsRef.current) wsRef.current.close();

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      // Deferred so an auto-connect from an effect never sets state synchronously.
      queueMicrotask(() => {
        setConnected(false);
        setStatus("Connection error");
      });
      return;
    }
    wsRef.current = ws;
    // Status updates flow from the socket's own (asynchronous) lifecycle events.
    ws.addEventListener("open", () => {
      ws.send(JSON.stringify({ role: "display" }));
      setConnected(true);
      setStatus("Connected");
    });

    ws.addEventListener("message", (event) => {
      try {
        const msg = JSON.parse(event.data) as FrameMessage;
        if (!msg.frame_b64 || !Array.isArray(msg.z)) return;
        const now = Date.now();
        setFrame(msg);
        setFrameSrc(decodeFrameSrc(msg));
        if (frameCounter.current.startedAt === 0) {
          frameCounter.current = { count: 0, startedAt: now };
        }
        frameCounter.current.count += 1;
        const elapsed = now - frameCounter.current.startedAt;
        if (elapsed > 1000) {
          setFps(Math.round((frameCounter.current.count * 1000) / elapsed));
          frameCounter.current = { count: 0, startedAt: now };
        }
      } catch {
        return;
      }
    });

    ws.addEventListener("close", () => {
      setConnected(false);
      setStatus("Disconnected");
    });

    ws.addEventListener("error", () => {
      setConnected(false);
      setStatus("Connection error");
    });
  }, [url]);

  // Auto-connect on session start; reconnect when the url changes while enabled.
  useEffect(() => {
    if (!enabled) return;
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [enabled, connect]);

  // Belt-and-suspenders close on unmount.
  useEffect(() => () => wsRef.current?.close(), []);

  return { url, setUrl, connected, status, frame, frameSrc, fps, connect, disconnect };
}
