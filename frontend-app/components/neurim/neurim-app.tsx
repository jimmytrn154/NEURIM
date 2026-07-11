"use client";

import { useSession } from "@/hooks/use-session";
import { Landing } from "@/components/neurim/landing";
import { SessionView } from "@/components/neurim/session-view";

/**
 * Top-level switch between the landing entry and the live session view.
 * Replaces the old neurim-dashboard monolith; rendered by app/page.tsx.
 */
export function NeurimApp() {
  const session = useSession();

  if (session.phase === "idle") {
    return <Landing onSubmit={session.startSession} isSubmitting={session.isSubmitting} />;
  }

  return <SessionView session={session} />;
}
