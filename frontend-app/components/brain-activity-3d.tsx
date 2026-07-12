"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { EEGFeatures } from "@/lib/neurim-types";
import { makeSyntheticEegFeatures } from "@/lib/mock-frame";
import { cn } from "@/lib/utils";

const fallbackPositions: Record<string, [number, number, number]> = {
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

function normalizeChannels(features?: EEGFeatures | null, reward = 0) {
  if (features?.channels?.length) {
    const powers = features.channels.map((c) => Math.log1p(Math.max(0, c.alpha_power ?? 0)));
    const max = Math.max(...powers, 1e-6);
    return features.channels.map((channel, index) => ({
      name: channel.name,
      position: channel.position ?? fallbackPositions[channel.name] ?? [0, 0, 0],
      intensity: Math.min(1, Math.max(0.06, powers[index] / max)),
    }));
  }
  return Object.keys(fallbackPositions).map((name) => ({
    name,
    position: fallbackPositions[name],
    intensity: 0.22 + Math.abs(reward) * 0.28,
  }));
}

function BrainMesh({ reward }: { reward: number }) {
  const left = useRef<THREE.Mesh>(null);
  const right = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const pulse = Math.sin(clock.elapsedTime * 1.8) * 0.015;
    if (left.current) left.current.scale.set(0.78 + pulse - reward * 0.025, 1.05 + pulse, 0.62);
    if (right.current) right.current.scale.set(0.78 + pulse + reward * 0.025, 1.05 + pulse, 0.62);
  });

  return (
    <group rotation={[0.15, 0, 0]}>
      <Sphere ref={left} args={[1, 48, 32]} position={[-0.36, 0, 0]} scale={[0.78, 1.05, 0.62]}>
        <meshStandardMaterial color="#2dd4e0" transparent opacity={0.2} roughness={0.85} />
      </Sphere>
      <Sphere ref={right} args={[1, 48, 32]} position={[0.36, 0, 0]} scale={[0.78, 1.05, 0.62]}>
        <meshStandardMaterial color="#f062a6" transparent opacity={0.2} roughness={0.85} />
      </Sphere>
    </group>
  );
}

function ElectrodeNodes({ features, reward }: { features?: EEGFeatures | null; reward: number }) {
  const channels = useMemo(() => normalizeChannels(features, reward), [features, reward]);
  const group = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    if (group.current) group.current.rotation.y = Math.sin(clock.elapsedTime * 0.25) * 0.12;
  });

  return (
    <group ref={group}>
      {channels.map((channel, index) => {
        const [x, y, z] = channel.position;
        const warm = channel.name === "F4" || x > 0;
        const color = warm ? new THREE.Color("#f062a6") : new THREE.Color("#43e4ee");
        const scale = 0.055 + channel.intensity * 0.075;
        return (
          <group key={channel.name} position={[x, y, z + 0.22]}>
            <Sphere args={[scale, 24, 16]}>
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.45 + channel.intensity} />
            </Sphere>
            <Sphere args={[scale * (1.6 + channel.intensity), 24, 16]}>
              <meshBasicMaterial color={color} transparent opacity={0.08 + channel.intensity * 0.13} />
            </Sphere>
            {index % 2 === 0 ? (
              <Sphere args={[0.012, 12, 8]} position={[-x * 0.18, -y * 0.18, -0.1]}>
                <meshBasicMaterial color={color} transparent opacity={0.2} />
              </Sphere>
            ) : null}
          </group>
        );
      })}
    </group>
  );
}

export function BrainActivity3D({
  features,
  reward,
  className,
}: {
  features?: EEGFeatures | null;
  reward: number;
  className?: string;
}) {
  const [syntheticFeatures, setSyntheticFeatures] = useState<EEGFeatures | null>(null);

  useEffect(() => {
    if (features?.channels?.length) {
      setSyntheticFeatures(null);
      return;
    }
    const update = () => {
      setSyntheticFeatures(
        makeSyntheticEegFeatures(reward, performance.now() / 1000, "brain-activity-3d"),
      );
    };
    update();
    const interval = window.setInterval(update, 160);
    return () => window.clearInterval(interval);
  }, [features, reward]);

  const displayFeatures = features?.channels?.length ? features : syntheticFeatures;

  return (
    <div className={cn("h-[330px] overflow-hidden rounded-md border bg-[#091013]", className)}>
      <Canvas camera={{ position: [0, 0.1, 3.7], fov: 42 }} dpr={[1, 1.75]}>
        <ambientLight intensity={0.55} />
        <pointLight position={[2, 2, 3]} intensity={1.6} color="#f062a6" />
        <pointLight position={[-2.4, 0.8, 2]} intensity={1.1} color="#2dd4e0" />
        <BrainMesh reward={reward} />
        <ElectrodeNodes features={displayFeatures} reward={reward} />
        <OrbitControls
          enablePan={false}
          enableZoom
          minDistance={2.1}
          maxDistance={5.4}
          autoRotate
          autoRotateSpeed={0.45}
        />
      </Canvas>
    </div>
  );
}
