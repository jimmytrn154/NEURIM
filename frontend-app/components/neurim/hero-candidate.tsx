import { cn } from "@/lib/utils";
import { stateBadge, type FrameState } from "@/lib/neurim-types";

export function HeroCandidate({
  frameSrc,
  state,
  fps,
  modelLabel = "SD-Turbo",
}: {
  frameSrc: string | null;
  state: FrameState;
  fps: number;
  modelLabel?: string;
}) {
  const isRecover = state === "recover";

  return (
    <div
      className={cn(
        "relative aspect-square overflow-hidden rounded-[18px]",
        "shadow-[0_26px_60px_rgba(16,24,40,.12),0_0_80px_rgba(63,98,246,.10)] dark:shadow-[0_40px_120px_rgba(0,0,0,.55),0_0_80px_rgba(63,98,246,.18)]",
      )}
    >
      {frameSrc ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={frameSrc}
          alt="Generated NEURIM candidate frame"
          className="h-full w-full object-cover"
        />
      ) : (
        <div className="relative flex h-full w-full items-center justify-center overflow-hidden bg-[#eef1f6] dark:bg-black">
          <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/25 to-transparent animate-scan" />
          <span className="font-mono text-xs text-muted-foreground">Rendering…</span>
        </div>
      )}

      <div
        className={cn(
          "absolute top-4 left-4 flex items-center gap-1.5 rounded-[20px] border px-3 py-1",
          "font-mono text-[10px] font-semibold uppercase tracking-wide",
          isRecover
            ? "text-[#e8b24a] bg-[rgba(232,178,74,.12)] border-[rgba(232,178,74,.35)]"
            : "text-[#3f62f6] bg-[rgba(63,98,246,.12)] border-[rgba(63,98,246,.3)] dark:text-[#dfe1e6] dark:bg-black/45 dark:border-white/10 dark:backdrop-blur",
        )}
      >
        <span className={cn(isRecover ? "text-[#e8b24a]" : "text-approach dark:text-approach-bright")}>
          ◆
        </span>
        <span>{stateBadge(state)}</span>
      </div>

      <div
        className={cn(
          "absolute top-4 right-4 rounded-[20px] border px-3 py-1",
          "font-mono text-[10px] font-medium uppercase tracking-wide",
          "text-[#475467] bg-[rgba(255,255,255,.8)] border-[rgba(16,24,40,.08)]",
          "dark:text-[#dfe1e6] dark:bg-black/45 dark:border-white/10 dark:backdrop-blur",
        )}
      >
        {modelLabel} · {fps} fps
      </div>
    </div>
  );
}
