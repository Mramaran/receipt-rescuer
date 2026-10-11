"""Training data: random patches of clean receipt scans, faded on the fly."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degrade import thermal_fade

EXTS = {".jpg", ".jpeg", ".png"}


def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS)


def split(paths, val_every=10):
    """Deterministic split: every `val_every`-th file (by sorted name) is held out."""
    val = [p for i, p in enumerate(paths) if i % val_every == val_every - 1]
    train = [p for i, p in enumerate(paths) if i % val_every != val_every - 1]
    return train, val


def read_gray(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img.astype(np.float32) / 255.0


class ReceiptPatches(Dataset):
    """Yields (faded, clean) patch pairs.

    With `fixed=True` the crop and the fading are seeded by the index, so the
    validation set is identical on every pass.
    """

    def __init__(self, paths, patch=192, length=10000, fixed=False, seed=0):
        self.paths, self.patch, self.length = list(paths), patch, length
        self.fixed, self.seed = fixed, seed

    def __len__(self):
        return self.length

    def _crop(self, img, rng):
        p = self.patch
        scale = float(rng.uniform(0.5, 1.0))
        if scale < 0.999:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape
        if h < p or w < p:
            img = cv2.copyMakeBorder(img, 0, max(0, p - h), 0, max(0, p - w), cv2.BORDER_REPLICATE)
            h, w = img.shape
        crop = None
        for _ in range(6):  # prefer patches that contain text
            y, x = int(rng.integers(0, h - p + 1)), int(rng.integers(0, w - p + 1))
            crop = img[y:y + p, x:x + p]
            if float((1.0 - crop).mean()) > 0.02:
                break
        return np.ascontiguousarray(crop)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i) if self.fixed else np.random.default_rng()
        path = self.paths[i % len(self.paths)] if self.fixed \
            else self.paths[int(rng.integers(0, len(self.paths)))]
        clean = self._crop(read_gray(path), rng)
        faded = thermal_fade(clean, rng)
        return torch.from_numpy(faded)[None], torch.from_numpy(clean)[None]
