"use client";

import { useSession } from "@/hooks/use-session";
import { EegGate } from "@/components/neurim/eeg-gate";
import { Landing } from "@/components/neurim/landing";
import { SessionView } from "@/components/neurim/session-view";

/**
 * Top-level switch between the landing entry and the live session view, gated on
 * the EPOC X connection lifecycle (see EegGate). The session only mounts once
 * the gate unlocks, so the frame stream never auto-connects before the headset
 * is ready. Replaces the old neurim-dashboard monolith; rendered by app/page.tsx.
 */
export function NeurimApp() {
  return (
    <EegGate>
      <NeurimSession />
    </EegGate>
  );
}

function NeurimSession() {
  const session = useSession();

  if (session.phase === "idle") {
    return <Landing onSubmit={session.startSession} isSubmitting={session.isSubmitting} />;
  }

  return <SessionView session={session} />;
}
