"use client";

import { TopBar } from "@/components/neurim/top-bar";
import { PromptBubble } from "@/components/neurim/prompt-bubble";
import { ProcessingState } from "@/components/neurim/processing-state";
import { HeroCandidate } from "@/components/neurim/hero-candidate";
import { SignalRail } from "@/components/neurim/signal-rail";
import { SteerInput } from "@/components/neurim/steer-input";
import type { NeurimSession } from "@/hooks/use-session";

/**
 * The 5a session layout: top bar → prompt bubble → processing beat OR the
 * two-column done state (hero candidate + signal rail) → bottom steer input.
 */
export function SessionView({ session }: { session: NeurimSession }) {
  const {
    phase,
    submittedPrompt,
    sessionId,
    statusText,
    connected,
    frame,
    frameSrc,
    fps,
    reward,
    showBrain,
    setShowBrain,
    wsUrl,
    setWsUrl,
    startSession,
    isSubmitting,
  } = session;

  const state = frame?.state ?? "explore";

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 pb-36 sm:px-6">
      <TopBar
        sessionId={sessionId}
        connected={connected}
        showBrain={showBrain}
        onToggleBrain={setShowBrain}
        wsUrl={wsUrl}
        onWsUrlChange={setWsUrl}
      />

      <div className="mt-10">
        <PromptBubble prompt={submittedPrompt} />
      </div>

      <div className="mt-10 flex-1">
        {phase === "processing" ? (
          <ProcessingState statusText={statusText} />
        ) : (
          <div className="flex flex-col items-stretch gap-6 md:flex-row">
            <div className="md:flex-[1.35]">
              <HeroCandidate frameSrc={frameSrc} state={state} fps={fps} />
            </div>
            <SignalRail
              reward={reward}
              state={state}
              features={frame?.eeg_features}
              showBrain={showBrain}
            />
          </div>
        )}
      </div>

      <div className="fixed inset-x-0 bottom-0 z-20 bg-gradient-to-t from-background via-background/95 to-transparent px-4 pb-6 pt-16">
        <div className="mx-auto max-w-3xl">
          <SteerInput onSubmit={startSession} disabled={isSubmitting} />
        </div>
      </div>
    </main>
  );
}
