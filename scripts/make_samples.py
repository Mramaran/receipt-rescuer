"""Write the sample receipts shown in the web app.

Each sample is a held-out scan (not used in training), faded with the same
generator, severity and seed as in the evaluation, so the app shows exactly
what was measured. Three were picked because they restore well; the fourth
because it does not.

    python scripts/make_samples.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rescuer.data import list_images, read_gray, split  # noqa: E402
from rescuer.degrade import thermal_fade  # noqa: E402
from rescuer.infer import prepare  # noqa: E402

# (scan name, fade severity, output file, label in the app)
SAMPLES = [
    ("X51007846366", 0.4, "sample1.png", "Sample 1"),
    ("X51005763940", 0.4, "sample2.png", "Sample 2"),
    ("X51006335547(2)", 0.6, "sample3.png", "Sample 3"),
    ("X51006554833", 0.8, "sample4.png", "Too faded to recover"),
]


def main():
    out = Path("samples")
    out.mkdir(exist_ok=True)
    _, val = split(list_images("data/sroie"))
    index = {p.stem: i for i, p in enumerate(val)}
    meta = []
    for name, severity, fname, label in SAMPLES:
        i = index[name]
        clean = prepare((read_gray(val[i]) * 255).astype(np.uint8))
        faded = thermal_fade(clean, np.random.default_rng(i), severity)  # seed 0 + position, as in evaluate.py
        cv2.imwrite(str(out / fname), (faded * 255).astype(np.uint8))
        meta.append({"file": fname, "label": label, "source": name, "severity": severity})
        print("wrote", out / fname)
    (out / "samples.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
