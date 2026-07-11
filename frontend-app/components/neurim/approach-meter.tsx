import { cn } from "@/lib/utils";

export function ApproachMeter({ reward }: { reward: number }) {
  const approaching = reward >= 0;
  const clamped = Math.min(1, Math.max(-1, reward));
  const pct = ((clamped + 1) / 2) * 100;

  return (
    <div>
      <div className="flex justify-between font-mono uppercase text-[10.5px] tracking-wide">
        <span className="text-[#7c6be0] dark:text-withdraw">withdraw</span>
        <span className="text-approach font-semibold">approach</span>
      </div>
      <div className="relative mt-2 h-[3px] rounded-full border border-border bg-[#e0e4ec] dark:bg-gradient-to-r dark:from-[#8b7cf0]/70 dark:via-[#2a2c33] dark:to-[#3f62f6]/80">
        <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-[#c2c7d2] dark:bg-white/10" />
        <div
          className={cn(
            "absolute top-1/2 h-[13px] w-[13px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary shadow-[0_0_0_4px_rgba(63,98,246,.18)]",
            approaching
              ? "dark:bg-approach dark:shadow-[0_0_0_4px_rgba(63,98,246,.25),0_0_16px_rgba(63,98,246,.5)]"
              : "dark:bg-withdraw dark:shadow-[0_0_0_4px_rgba(139,124,240,.25),0_0_16px_rgba(139,124,240,.5)]"
          )}
          style={{ left: pct + "%" }}
        />
      </div>
    </div>
  );
}
