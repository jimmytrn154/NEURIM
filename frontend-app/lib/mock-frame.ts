// Deterministic mock-frame generation for offline / fallback mode. Extracted
// verbatim from the original neurim-dashboard monolith so the redesign stays
// reviewable with no backend running.

import type { EEGFeatures, FrameMessage } from "@/lib/neurim-types";

export const examplePrompts = [
  "A calm golden retriever puppy on white bedding",
  "A futuristic bioluminescent garden at dawn",
  "A soft cinematic portrait with warm studio light",
];

export const epocPositions: Record<string, [number, number, number]> = {
  AF3: [-0.42, 0.88, 0.22],
  F7: [-0.86, 0.58, 0.04],
  F3: [-0.46, 0.55, 0.36],
  FC5: [-0.72, 0.22, 0.22],
  T7: [-0.95, -0.08, 0],
  P7: [-0.78, -0.58, 0.1],
  O1: [-0.34, -0.9, 0.18],
  O2: [0.34, -0.9, 0.18],
  P8: [0.78, -0.58, 0.1],
  T8: [0.95, -0.08, 0],
  FC6: [0.72, 0.22, 0.22],
  F4: [0.46, 0.55, 0.36],
  F8: [0.86, 0.58, 0.04],
  AF4: [0.42, 0.88, 0.22],
};

function promptHash(input: string) {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededUnit(seed: number, index: number) {
  const value = Math.sin(seed * 0.00001 + index * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

export function makeSyntheticEegFeatures(
  reward: number,
  t: number,
  seedSource = "live-signal",
): EEGFeatures {
  const seed = promptHash(seedSource);
  const clippedReward = Math.max(-1, Math.min(1, reward));
  const phase = t * 1.9;

  const channels = Object.entries(epocPositions).map(([name, position], index) => {
    const lateral = position[0] >= 0 ? 1 : -1;
    const rhythmic = 0.5 + 0.5 * Math.sin(phase + index * 0.63 + seededUnit(seed, index) * Math.PI * 2);
    const shimmer = 0.5 + 0.5 * Math.sin(phase * 2.3 + index * 0.41);
    const alphaPower = Math.max(
      0.08,
      0.18
        + rhythmic * 1.05
        + shimmer * 0.34
        + Math.max(0, clippedReward * lateral) * 0.52
        + Math.abs(clippedReward) * 0.14,
    );
    const signedValue =
      lateral * clippedReward * 5.2
      + Math.sin(phase * 1.35 + index * 0.82) * 2.6
      + (seededUnit(seed, index + 64) - 0.5) * 1.6;
    const quality = Math.max(
      0.58,
      Math.min(0.98, 0.76 + Math.sin(phase * 0.4 + index * 0.35) * 0.08),
    );
    return {
      name,
      value: Number(signedValue.toFixed(3)),
      alpha_power: Number(alphaPower.toFixed(3)),
      quality: Number(quality.toFixed(3)),
      position,
    };
  });

  return {
    channels,
    faa: {
      raw: Number((clippedReward * 0.3 + Math.sin(phase * 0.7) * 0.05).toFixed(3)),
      reward: Number(clippedReward.toFixed(3)),
      left_channel: "F3",
      right_channel: "F4",
    },
  };
}

function makeMockImageSrc(seed: number) {
  const hueA = Math.round(8 + seededUnit(seed, 1) * 36);
  const hueB = Math.round(178 + seededUnit(seed, 2) * 54);
  const hueC = Math.round(248 + seededUnit(seed, 3) * 42);
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 720">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="hsl(${hueA} 92% 68%)"/>
          <stop offset="50%" stop-color="hsl(${hueB} 70% 50%)"/>
          <stop offset="100%" stop-color="hsl(${hueC} 76% 60%)"/>
        </linearGradient>
        <radialGradient id="light" cx="68%" cy="18%" r="60%">
          <stop offset="0%" stop-color="white" stop-opacity="0.62"/>
          <stop offset="100%" stop-color="white" stop-opacity="0"/>
        </radialGradient>
        <filter id="soften">
          <feGaussianBlur stdDeviation="20"/>
        </filter>
      </defs>
      <rect width="720" height="720" rx="42" fill="url(#bg)"/>
      <rect width="720" height="720" rx="42" fill="url(#light)"/>
      <ellipse cx="300" cy="360" rx="190" ry="150" fill="white" opacity="0.18" filter="url(#soften)"/>
      <ellipse cx="500" cy="460" rx="160" ry="110" fill="black" opacity="0.10" filter="url(#soften)"/>
      <path d="M100 528 C230 410 322 594 462 476 C522 426 588 410 642 438" fill="none" stroke="white" stroke-width="18" stroke-linecap="round" opacity="0.28"/>
    </svg>
  `;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

export type MockSession = {
  candidateSrc: string;
  frame: FrameMessage;
};

export function makeMockSession(prompt: string, t: number): MockSession {
  const seed = promptHash(prompt);
  const reward = Number((-0.08 + seededUnit(seed, 8) * 0.72).toFixed(3));
  const rawFaa = Number((reward * 0.22 + (seededUnit(seed, 9) - 0.5) * 0.04).toFixed(3));
  const z = Array.from({ length: 8 }, (_, index) =>
    Number((seededUnit(seed, index + 16) * 2 - 1).toFixed(3))
  );
  const synthetic = makeSyntheticEegFeatures(reward, t, prompt);

  return {
    candidateSrc: makeMockImageSrc(seed),
    frame: {
      frame_b64: "mock",
      z,
      step_index: 1,
      t,
      format: "mock",
      state: "explore",
      reward_estimate: reward,
      eeg_features: {
        ...synthetic,
        faa: {
          ...synthetic.faa,
          raw: rawFaa,
        },
      },
    },
  };
}
