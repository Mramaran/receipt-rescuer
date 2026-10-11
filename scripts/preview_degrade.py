"""Write side-by-side previews: clean | light fade | medium | heavy.

Use it to compare the synthetic fading against photos of real faded receipts.

    python scripts/preview_degrade.py --data data/sroie --n 6
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rescuer.data import list_images, read_gray  # noqa: E402
from rescuer.degrade import thermal_fade  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sroie")
    ap.add_argument("--out", default="outputs/preview")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    paths = list_images(args.data)
    for path in [paths[i] for i in rng.choice(len(paths), size=min(args.n, len(paths)), replace=False)]:
        clean = read_gray(path)
        h = min(clean.shape[0], 900)
        clean = clean[:h]
        row = [clean] + [thermal_fade(clean, rng, s) for s in (0.3, 0.6, 0.9)]
        cv2.imwrite(str(out / f"{path.stem}.png"), (np.hstack(row) * 255).astype(np.uint8))
        print("wrote", out / f"{path.stem}.png")


if __name__ == "__main__":
    main()
