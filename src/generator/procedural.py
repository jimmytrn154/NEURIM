"""CPU-only fallback renderer: a deterministic function of z, no GPU or model
weights required. This is what proves the fake-reward loop end-to-end (build
order step 1), and it's the safety net if real-time text-to-3D blows the
latency budget on demo day.

z's first few dimensions are mapped onto a stylized little brown pup's coat,
pose, ear shape, tail lift, and scale, so the demo still stays dog-shaped when
the OpenAI image backend is unavailable.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


def _dim(z: np.ndarray, i: int, default: float = 0.0) -> float:
    return float(z[i]) if i < len(z) else default


class ProceduralRenderer:
    def render(self, z: np.ndarray, size: int = 512) -> Image.Image:
        z = np.clip(z, -1.0, 1.0)
        coat_t = (_dim(z, 0) + 1.0) / 2.0
        ear_t = (_dim(z, 1) + 1.0) / 2.0
        tail_t = (_dim(z, 2) + 1.0) / 2.0
        tilt = _dim(z, 3) * 7.0
        scale = 0.86 + _dim(z, 4) * 0.08

        dark = np.array([91, 52, 30])
        light = np.array([188, 113, 55])
        coat = tuple(np.round(dark + coat_t * (light - dark)).astype(int))
        shadow = tuple(max(0, int(c * 0.72)) for c in coat)
        highlight = tuple(min(255, int(c * 1.22)) for c in coat)
        tan = (218, 166, 104)

        img = Image.new("RGB", (size, size), (10, 10, 16))
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = size / 2, size / 2
        s = size / 512 * scale

        def xy(x: float, y: float) -> tuple[float, float]:
            return cx + x * s, cy + y * s

        def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
            ax, ay = xy(x0, y0)
            bx, by = xy(x1, y1)
            return ax, ay, bx, by

        # Tail and rear silhouette.
        tail_angle = math.radians(-145 + tail_t * 70)
        tail_base = xy(-124, 26)
        tail_tip = (tail_base[0] + math.cos(tail_angle) * 105 * s, tail_base[1] + math.sin(tail_angle) * 105 * s)
        draw.line([tail_base, tail_tip], fill=(*shadow, 255), width=max(8, int(18 * s)))
        draw.ellipse((tail_tip[0] - 12 * s, tail_tip[1] - 12 * s, tail_tip[0] + 12 * s, tail_tip[1] + 12 * s), fill=(*shadow, 255))

        # Body, paws, and head.
        draw.ellipse(box(-145, -45, 80, 105), fill=(*coat, 255))
        draw.ellipse(box(-105, 62, -58, 148), fill=(*shadow, 255))
        draw.ellipse(box(-22, 62, 26, 150), fill=(*shadow, 255))
        draw.ellipse(box(-117, 132, -43, 162), fill=(*shadow, 255))
        draw.ellipse(box(-37, 132, 43, 162), fill=(*shadow, 255))

        draw.ellipse(box(42, -94, 172, 38), fill=(*coat, 255))
        draw.ellipse(box(94, -30, 190, 45), fill=(*coat, 255))
        draw.ellipse(box(111, -5, 181, 52), fill=(*tan, 255))

        # Floppy ears vary from compact to long.
        ear_drop = 78 + ear_t * 48
        draw.polygon([xy(55, -76), xy(6, -39), xy(39, -76 + ear_drop)], fill=(*shadow, 255))
        draw.polygon([xy(116, -91), xy(160, -54), xy(130, -78 + ear_drop * 0.82)], fill=(*shadow, 255))

        # Face details.
        draw.ellipse(box(122, -55, 139, -38), fill=(18, 14, 12, 255))
        draw.ellipse(box(127, -52, 133, -46), fill=(255, 246, 220, 255))
        draw.ellipse(box(160, -11, 183, 9), fill=(18, 14, 12, 255))
        draw.arc(box(141, 5, 174, 33), start=10, end=150, fill=(42, 23, 18, 255), width=max(1, int(3 * s)))

        # Soft chest and forehead highlights keep the pup readable on the dark stage.
        draw.ellipse(box(-8, -4, 70, 90), fill=(*highlight, 135))
        draw.ellipse(box(78, -70, 126, -25), fill=(*highlight, 120))

        if abs(tilt) > 0.1:
            layer = layer.rotate(tilt, resample=Image.Resampling.BICUBIC, center=(cx, cy))
        background = Image.new("RGBA", (size, size), (*img.getpixel((0, 0)), 255))
        img.paste(Image.alpha_composite(background, layer).convert("RGB"))
        return img
