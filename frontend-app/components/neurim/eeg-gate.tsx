"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Brain, RefreshCw, WifiOff, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { useEegStatus, type EegStatusHook } from "@/hooks/use-eeg-status";
import { ThemeToggle } from "@/components/neurim/theme-toggle";

/**
 * Gates the app on the EPOC X connection lifecycle.
 *
 * - Backend unreachable (no local api_server.py) → pass straight through so the
 *   existing offline/mock experience keeps working; the pill reads "offline".
 * - Headset not connected → a retry screen.
 * - Connecting / calibrating → an animated waiting screen; calibration runs a
 *   countdown synced to the backend's reported `calibration_seconds`.
 * - Ready (connected + calibrated) → renders the app ("loads up the chat").
 *
 * Once we've unlocked (ready or offline) we stay unlocked for the page load so a
 * mid-session blip never yanks the user out of the chat — the pill surfaces it
 * instead.
 */
export function EegGate({ children }: { children: ReactNode }) {
  const eeg = useEegStatus();
  const [unlocked, setUnlocked] = useState(false);

  const state = eeg.status?.state;
  const ready = eeg.reachable && state === "ready";
  const offline = !eeg.reachable && !eeg.loading;

  // Sticky latch: once ready (or offline/mock) we stay unlocked for the page
  // load so a mid-session blip never yanks the user out of the chat. Adjusting
  // state during render is the sanctioned pattern here and converges in one pass.
  if (!unlocked && (ready || offline)) {
    setUnlocked(true);
  }

  if (!unlocked) {
    return <EegGateScreen eeg={eeg} onSkip={() => setUnlocked(true)} />;
  }

  return (
    <>
      <EegStatusPill eeg={eeg} />
      {children}
    </>
  );
}

// ---- Full-screen gate ------------------------------------------------------

function EegGateScreen({ eeg, onSkip }: { eeg: EegStatusHook; onSkip: () => void }) {
  const { status, loading, retrying, retry } = eeg;
  const state = status?.state ?? "unknown";

  // `retrying` and the optimistic "connecting" both land on the connecting view.
  const connecting = retrying || state === "connecting" || state === "connected";
  const calibrating = state === "calibrating";

  return (
    <div className="min-h-screen relative bg-landing-bg overflow-hidden">
      <div className="absolute top-0 left-0 right-0 px-6 py-[18px] flex items-center z-20">
        <div className="flex items-center gap-2">
          <Brain size={20} className="text-approach" />
          <span className="font-sans font-semibold text-[16px] tracking-[-0.01em] text-foreground">
            NEURIM
          </span>
        </div>
      </div>

      <div className="absolute inset-0 flex flex-col items-center justify-center px-6 text-center">
        {/* Soft blue glow behind the card, matching the landing hero. */}
        <div
          className="pointer-events-none absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[560px] h-[280px]"
          style={{
            background:
              "radial-gradient(ellipse 55% 55% at 50% 50%, rgba(63,98,246,.30), rgba(63,98,246,.10) 45%, transparent 72%)",
            filter: "blur(26px)",
          }}
        />

        <div className="relative w-full max-w-[440px] rounded-[24px] border border-border bg-card px-8 py-9 shadow-[0_10px_40px_rgba(16,24,40,.12)] dark:shadow-[0_30px_90px_rgba(0,0,0,.5)]">
          {loading && !status ? (
            <CheckingView />
          ) : calibrating ? (
            <CalibratingView status={status!} />
          ) : connecting ? (
            <ConnectingView />
          ) : (
            <DisconnectedView eeg={eeg} onRetry={retry} onSkip={onSkip} />
          )}
        </div>
      </div>

      <ThemeToggle className="fixed bottom-5 right-5 z-30 h-10 w-10 rounded-full border border-border bg-card/80 backdrop-blur shadow-sm" />
    </div>
  );
}

function StatusHeading({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <>
      <h2 className="font-serif text-[26px] leading-tight text-foreground">{title}</h2>
      <p className="mt-2 text-[14px] text-muted-foreground">{subtitle}</p>
    </>
  );
}

function CheckingView() {
  return (
    <div className="flex flex-col items-center">
      <BrainOrb pulsing />
      <div className="mt-6">
        <StatusHeading title="Checking EPOC X" subtitle="Reading the headset connection…" />
      </div>
    </div>
  );
}

function ConnectingView() {
  return (
    <div className="flex flex-col items-center">
      <BrainOrb pulsing />
      <div className="mt-6">
        <StatusHeading title="Connecting to EPOC X" subtitle="Establishing the Cortex link…" />
      </div>
      <div className="mt-6 flex items-center gap-1.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-approach animate-dot"
            style={{ animationDelay: `${i * 160}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

function CalibratingView({ status }: { status: NonNullable<EegStatusHook["status"]> }) {
  const total = status.calibration_seconds > 0 ? status.calibration_seconds : 20;
  const startMs = status.last_connected_at ? Date.parse(status.last_connected_at) : NaN;

  // Client-only subview (never SSR'd — the gate renders CheckingView on the
  // server), so seeding from Date.now() in the initializer is safe.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 200);
    return () => clearInterval(id);
  }, []);

  const elapsed = Number.isFinite(startMs) ? Math.max(0, (now - startMs) / 1000) : 0;
  // Hold shy of 100% until the backend actually reports `ready`.
  const pct = Math.min(96, (elapsed / total) * 100);
  const remaining = Math.max(0, Math.ceil(total - elapsed));

  return (
    <div className="flex flex-col items-center">
      <BrainOrb pulsing />
      <div className="mt-6">
        <StatusHeading
          title="Calibrating baseline"
          subtitle="Relax and hold still while we set your resting signal."
        />
      </div>

      <div className="mt-7 w-full">
        <div className="relative h-2 w-full overflow-hidden rounded-full bg-secondary">
          <div
            className="h-full rounded-full bg-approach transition-[width] duration-200 ease-out"
            style={{ width: `${pct}%` }}
          />
          <div className="pointer-events-none absolute inset-y-0 left-0 w-1/3 animate-sweep bg-gradient-to-r from-transparent via-white/40 to-transparent" />
        </div>
        <div className="mt-2 flex justify-between font-mono text-[11px] uppercase tracking-wide text-muted-foreground">
          <span>Calibrating</span>
          <span>{remaining > 0 ? `${remaining}s left` : "Finishing up…"}</span>
        </div>
      </div>
    </div>
  );
}

function DisconnectedView({
  eeg,
  onRetry,
  onSkip,
}: {
  eeg: EegStatusHook;
  onRetry: () => void;
  onSkip: () => void;
}) {
  const { status, retrying } = eeg;
  const reachable = eeg.reachable;
  const isError = status?.state === "error";

  const subtitle = !reachable
    ? "Can't reach the local api_server.py. Start it, then retry."
    : status?.last_error
      ? status.last_error
      : "Put on the EPOC X headset and make sure Cortex is running, then retry.";

  return (
    <div className="flex flex-col items-center">
      <div className="grid h-16 w-16 place-items-center rounded-full bg-secondary text-muted-foreground">
        <WifiOff size={26} />
      </div>
      <div className="mt-6">
        <StatusHeading
          title={isError ? "EPOC X connection failed" : "EPOC X not connected"}
          subtitle={subtitle}
        />
      </div>

      {reachable && status?.next_retry_at ? (
        <AutoRetryHint nextRetryAt={status.next_retry_at} />
      ) : null}

      <button
        type="button"
        onClick={onRetry}
        disabled={retrying}
        className="mt-7 inline-flex items-center gap-2 rounded-full bg-primary px-6 py-2.5 text-[14px] font-medium text-primary-foreground transition hover:bg-primary/90 disabled:opacity-60"
      >
        <RefreshCw size={16} className={cn(retrying && "animate-spin")} />
        {retrying ? "Retrying…" : "Retry connection"}
      </button>

      <button
        type="button"
        onClick={onSkip}
        className="mt-4 text-[12px] text-muted-foreground underline-offset-4 transition hover:text-foreground hover:underline"
      >
        Continue without headset
      </button>
    </div>
  );
}

function AutoRetryHint({ nextRetryAt }: { nextRetryAt: string }) {
  const targetMs = Date.parse(nextRetryAt);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);

  if (!Number.isFinite(targetMs)) return null;
  const secs = Math.max(0, Math.ceil((targetMs - now) / 1000));
  return (
    <p className="mt-4 font-mono text-[11px] uppercase tracking-wide text-muted-foreground">
      {secs > 0 ? `Auto-retrying in ${secs}s` : "Retrying…"}
    </p>
  );
}

function BrainOrb({ pulsing }: { pulsing?: boolean }) {
  return (
    <div className="relative grid h-16 w-16 place-items-center">
      <span
        className={cn(
          "absolute inset-0 rounded-full bg-approach/20",
          pulsing && "animate-breathe"
        )}
      />
      <span className="absolute inset-2 rounded-full bg-approach/25" />
      <Brain size={26} className="relative text-approach" />
    </div>
  );
}

// ---- Persistent top-of-app status pill -------------------------------------

function EegStatusPill({ eeg }: { eeg: EegStatusHook }) {
  const { status, reachable, retrying, retry } = eeg;
  const state = reachable ? status?.state ?? "unknown" : "offline";

  let dot = "bg-approach animate-blink";
  let label = "EPOC X connected";
  let showRetry = false;

  switch (state) {
    case "ready":
      dot = "bg-approach animate-blink";
      label = "EPOC X connected";
      break;
    case "connecting":
    case "connected":
      dot = "bg-amber-500 animate-blink";
      label = "Reconnecting…";
      break;
    case "calibrating":
      dot = "bg-amber-500 animate-blink";
      label = "Calibrating…";
      break;
    case "disconnected":
    case "error":
      dot = "bg-destructive";
      label = "EPOC X disconnected";
      showRetry = true;
      break;
    case "offline":
      dot = "bg-muted-foreground";
      label = "EPOC X offline · mock";
      break;
    default:
      dot = "bg-muted-foreground";
      label = "EPOC X…";
  }

  return (
    <div className="pointer-events-none fixed left-1/2 top-3 z-40 -translate-x-1/2">
      <div className="pointer-events-auto flex items-center gap-2 rounded-full border border-border bg-card/85 px-3 py-1.5 shadow-sm backdrop-blur">
        <span className={cn("h-[7px] w-[7px] rounded-full", dot)} />
        <span className="font-mono text-[11px] uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        {state === "ready" ? <Check size={12} className="text-approach" /> : null}
        {showRetry ? (
          <button
            type="button"
            onClick={retry}
            disabled={retrying}
            className="ml-0.5 inline-flex items-center gap-1 rounded-full bg-primary px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-primary-foreground transition hover:bg-primary/90 disabled:opacity-60"
          >
            <RefreshCw size={10} className={cn(retrying && "animate-spin")} />
            Retry
          </button>
        ) : null}
      </div>
    </div>
  );
}
