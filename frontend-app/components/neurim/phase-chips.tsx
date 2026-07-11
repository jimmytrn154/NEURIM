import { cn } from "@/lib/utils";
import { PHASES, type FrameState } from "@/lib/neurim-types";

export function PhaseChips({ state }: { state: FrameState }) {
  return (
    <div className="flex flex-wrap gap-2">
      {PHASES.map((phase) => {
        const active = phase.key === state;
        return (
          <span
            key={phase.key}
            className={cn(
              "font-mono uppercase text-[11px] tracking-wide rounded-full px-3 py-1.5 border",
              active
                ? "bg-primary text-white border-transparent shadow-[0_0_16px_rgba(63,98,246,0.5)]"
                : "bg-card dark:bg-secondary border-border text-[#98a2b3] dark:text-muted-foreground"
            )}
          >
            {phase.label}
          </span>
        );
      })}
      {state === "recover" ? (
        <span className="font-mono uppercase text-[11px] tracking-wide rounded-full px-3 py-1.5 border border-[#e8b24a]/50 text-[#e8b24a] bg-[#e8b24a]/10 shadow-[0_0_16px_rgba(232,178,74,0.35)]">
          Recover
        </span>
      ) : null}
    </div>
  );
}
