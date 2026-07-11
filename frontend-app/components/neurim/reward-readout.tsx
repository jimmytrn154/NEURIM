import { cn } from "@/lib/utils";

export function RewardReadout({ reward }: { reward: number }) {
  const approaching = reward >= 0;
  const formatted = `${approaching ? "+" : "−"}${Math.abs(reward).toFixed(2)}`;

  return (
    <div>
      <div className="font-serif text-[13px] text-muted-foreground">
        Reward r(t) ·{" "}
        <span
          className={cn(
            approaching ? "text-approach dark:text-approach-bright" : "text-[#7c6be0] dark:text-withdraw"
          )}
        >
          {approaching ? "approach" : "withdraw"}
        </span>
      </div>
      <div
        className={cn(
          "mt-2 font-serif text-[66px] leading-none tracking-[-.02em]",
          approaching
            ? "text-approach dark:text-approach-bright dark:[text-shadow:0_0_30px_rgba(63,98,246,0.45)]"
            : "text-[#7c6be0] dark:text-withdraw dark:[text-shadow:0_0_30px_rgba(139,124,240,0.4)]"
        )}
      >
        {formatted}
      </div>
    </div>
  );
}
