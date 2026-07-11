// Shared NEURIM message + API types. These mirror src/common/messages.py and are
// carried over verbatim from the original neurim-dashboard monolith.

export type EEGFeatures = {
  channels: Array<{
    name: string;
    value: number;
    alpha_power?: number;
    quality?: number;
    position: [number, number, number];
  }>;
  faa: {
    raw: number | null;
    reward: number;
    left_channel: string;
    right_channel: string;
  };
};

export type FrameState =
  | "calibrate"
  | "explore"
  | "refine"
  | "settle"
  | "recover"
  | (string & {});

export type FrameMessage = {
  frame_b64: string;
  z: number[];
  step_index: number;
  t: number;
  format: "jpeg" | "png" | string;
  state: FrameState;
  reward_estimate: number;
  eeg_features?: EEGFeatures | null;
};

export type BackendSession = {
  running: boolean;
  pid: number | null;
  started_at: string | null;
  prompt: string | null;
  exit_code: number | null;
};

export type SessionIntentResponse = {
  ok: boolean;
  session_id: string;
  prompt: string;
  max_steps: number;
  prompt_routing: "local_api_server" | string;
  render_contract: string;
  backend_url: string;
  backend_session: BackendSession;
};

// The four nominal 5a phases. `recover` is the escape state, rendered distinctly
// rather than as a fifth chip.
export const PHASES = [
  { key: "calibrate", label: "Calibrate" },
  { key: "explore", label: "Explore" },
  { key: "refine", label: "Refine" },
  { key: "settle", label: "Settle" },
] as const;

export type PhaseKey = (typeof PHASES)[number]["key"];

// Maps a frame state to the badge shown top-left of the hero candidate.
export function stateBadge(state: FrameState): string {
  switch (state) {
    case "calibrate":
      return "CALIBRATE";
    case "explore":
      return "EXPLORE";
    case "refine":
      return "REFINE";
    case "settle":
      return "SETTLE";
    case "recover":
      return "RECOVER";
    default:
      return String(state || "").toUpperCase();
  }
}

export function decodeFrameSrc(msg: FrameMessage): string {
  return `data:image/${msg.format || "jpeg"};base64,${msg.frame_b64}`;
}
