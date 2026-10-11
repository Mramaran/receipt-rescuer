"""Synthetic thermal-receipt fading.

Thermal paper does not fade evenly: ink density drops in patches, faint strokes
vanish first, weak print-head dots leave vertical streaks, and the paper itself
greys or yellows. This module models those effects on a clean scan so that a
restoration network can be trained on (faded, clean) pairs.

All images are float32, HxW, in [0, 1], dark ink on light paper.
"""
import cv2
import numpy as np


def _field(shape, cell, rng):
    """Smooth random field in [0, 1] with features about `cell` pixels wide."""
    h, w = shape
    gh, gw = max(2, h // cell + 2), max(2, w // cell + 2)
    small = rng.random((gh, gw), dtype=np.float32)
    return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)


def _jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", np.clip(img * 255, 0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def thermal_fade(clean, rng, severity=None):
    """Return a faded version of `clean`.

    severity in [0, 1]: 0 is a light fade, 1 leaves a few percent of the ink in
    the worst patches. Drawn at random when None.
    """
    h, w = clean.shape
    s = float(rng.uniform(0.15, 1.0)) if severity is None else float(severity)

    paper = max(float(np.percentile(clean, 95)), 0.5)
    ink = np.clip(1.0 - clean / paper, 0, 1)

    # How much ink survives: an overall level, lowered further in patches.
    base = float(np.interp(s, [0, 1], [0.60, 0.06]))
    patch = _field((h, w), int(rng.integers(48, 160)), rng)
    patchiness = float(rng.uniform(0.3, 1.0)) * s
    keep = base * np.exp(-2.5 * patchiness * patch)

    # Weak print-head dots: vertical streaks with less ink.
    for _ in range(int(rng.integers(0, 4))):
        x0 = int(rng.integers(0, w))
        wd = int(rng.integers(2, 12))
        keep[:, x0:x0 + wd] *= float(rng.uniform(0.2, 0.8))

    # One side exposed to light or heat: a ramp across the paper.
    if rng.random() < 0.5:
        ramp = np.linspace(0, 1, w, dtype=np.float32)[None, :] if rng.random() < 0.5 \
            else np.linspace(0, 1, h, dtype=np.float32)[:, None]
        if rng.random() < 0.5:
            ramp = 1.0 - ramp
        keep = keep * (1.0 - float(rng.uniform(0, 0.6)) * s * ramp)

    # Faint strokes disappear before bold ones.
    gamma = float(rng.uniform(1.0, 1.6))
    faded_ink = (ink ** gamma) * keep

    # Paper tone drifts and is uneven.
    tone = float(rng.uniform(0.78, 0.97))
    tone_map = tone + 0.06 * (_field((h, w), int(rng.integers(64, 256)), rng) - 0.5)
    out = tone_map * (1.0 - faded_ink)

    # Phone capture: shadow, slight blur, sensor noise, JPEG.
    shadow = 1.0 - float(rng.uniform(0, 0.25)) * _field((h, w), int(rng.integers(96, 320)), rng)
    out = out * shadow
    if rng.random() < 0.7:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 1.3)))
    out = out + rng.normal(0, float(rng.uniform(0.005, 0.04)), out.shape).astype(np.float32)
    out = np.clip(out, 0, 1).astype(np.float32)
    if rng.random() < 0.7:
        out = _jpeg(out, rng.integers(35, 90))
    return out


def clahe_baseline(img):
    """Classical contrast fix a scanner app would apply. Used as the baseline."""
    u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u8)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.float32) / 255.0
