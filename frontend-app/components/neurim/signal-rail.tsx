"use client";

import dynamic from "next/dynamic";
import { RewardReadout } from "@/components/neurim/reward-readout";
import { ApproachMeter } from "@/components/neurim/approach-meter";
import { PhaseChips } from "@/components/neurim/phase-chips";
import { Skeleton } from "@/components/ui/skeleton";
import type { EEGFeatures, FrameState } from "@/lib/neurim-types";

const BrainActivity3D = dynamic(
  () => import("@/components/brain-activity-3d").then((mod) => mod.BrainActivity3D),
  {
    ssr: false,
    loading: () => <Skeleton className="h-[300px] rounded-[20px] bg-[#0A0B12]" />,
  }
);

/**
 * The 5a signal rail: reward readout + bipolar meter, phase chips, and the 3D
 * brain pinned to the bottom (shown only when "Alpha asymmetry" is on).
 */
export function SignalRail({
  reward,
  state,
  features,
  showBrain,
}: {
  reward: number;
  state: FrameState;
  features?: EEGFeatures | null;
  showBrain: boolean;
}) {
  return (
    <div className="flex flex-1 flex-col gap-6 rounded-[28px] border border-border bg-card/50 p-6">
      <div className="flex flex-col gap-5">
        <RewardReadout reward={reward} />
        <ApproachMeter reward={reward} />
      </div>

      <PhaseChips state={state} />

      {showBrain ? (
        <div className="mt-auto overflow-hidden rounded-[14px] border border-[rgba(16,24,40,.1)] bg-[#0A0B12] dark:border-white/8">
          <div className="flex items-center justify-between px-4 pt-3">
            <span className="font-sans font-semibold text-[11px] text-[#eceef2]">Electrode scan</span>
            <span className="font-mono font-medium text-[9px] text-[#aab2d8] bg-white/[.08] rounded-[10px] px-2 py-0.5">
              14ch
            </span>
          </div>
          <BrainActivity3D
            features={features}
            reward={reward}
            className="h-[300px] rounded-none border-0 bg-transparent"
          />
        </div>
      ) : null}
    </div>
  );
}
