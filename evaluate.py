"""Measure what restoration does to OCR, and whether the flags mean anything.

Synthetic mode (held-out clean scans, faded with a fixed seed):
    python evaluate.py --weights runs/run1/best.pt --clean data/sroie --n 40 --severity 0.8
  The reference text is the OCR reading of the clean scan, so the numbers say
  how much of what OCR can read on a clean receipt is recovered after fading.

Real mode (photos of faded receipts with hand transcripts <name>.txt beside them):
    python evaluate.py --weights runs/run1/best.pt --real data/real

Reports character error rate for the faded input, a classical CLAHE contrast
fix, and the restored image, plus the error rate of flagged versus unflagged
characters. Results go to outputs/eval/<name>.json.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from rescuer.data import list_images, read_gray, split
from rescuer.degrade import clahe_baseline, thermal_fade
from rescuer.flag import CONF_THRESH
from rescuer.infer import prepare
from rescuer.metrics import cer, flag_counts
from rescuer.pipeline import Rescuer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clean", help="folder of clean scans (synthetic mode)")
    ap.add_argument("--real", help="folder of real photos with <name>.txt transcripts")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--severity", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="skip this many held-out scans first")
    ap.add_argument("--conf-thresh", type=float, default=CONF_THRESH)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if bool(args.clean) == bool(args.real):
        raise SystemExit("give exactly one of --clean or --real")

    rescuer = Rescuer(args.weights)
    if rescuer.ocr is None:
        raise SystemExit(rescuer.ocr_error)

    items = []  # (name, faded image, reference text)
    if args.clean:
        _, val = split(list_images(args.clean))
        for i, path in enumerate(val[args.skip:args.skip + args.n], start=args.skip):
            clean = prepare((read_gray(path) * 255).astype(np.uint8))
            faded = thermal_fade(clean, np.random.default_rng(args.seed + i), args.severity)
            items.append((path.stem, faded, rescuer.ocr.read(clean).text))
        mode = f"synthetic severity={args.severity} seed={args.seed} skip={args.skip}"
    else:
        for path in list_images(args.real)[:args.n]:
            txt = path.with_suffix(".txt")
            if not txt.exists():
                print("skip (no transcript):", path.name)
                continue
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            items.append((path.stem, prepare(img), txt.read_text(encoding="utf-8")))
        mode = "real photos with hand transcripts"
    if not items:
        raise SystemExit("nothing to evaluate")

    tot = {k: 0 for k in ("ref", "faded", "clahe", "restored", "flag_n", "flag_wrong", "clear_n",
                          "clear_wrong", "ens_n", "ens_wrong", "conf_n", "conf_wrong")}
    rows, t0 = [], time.time()
    for name, faded, ref in items:
        e_f, n_ref = cer(ref, rescuer.ocr.read(faded).text)
        e_c, _ = cer(ref, rescuer.ocr.read(clahe_baseline(faded)).text)
        res = rescuer.run(faded, conf_thresh=args.conf_thresh)
        e_r, _ = cer(ref, res["reading"].text)
        fc = flag_counts(res["reading"].text, res["flags"], ref)
        rows.append({"name": name, "ref_chars": n_ref, "cer_faded": e_f / max(n_ref, 1),
                     "cer_clahe": e_c / max(n_ref, 1), "cer_restored": e_r / max(n_ref, 1), **fc})
        tot["ref"] += n_ref
        tot["faded"] += e_f
        tot["clahe"] += e_c
        tot["restored"] += e_r
        for k in fc:
            tot[k] += fc[k]
        print(f"{name}: faded {rows[-1]['cer_faded']:.3f}  clahe {rows[-1]['cer_clahe']:.3f}  "
              f"restored {rows[-1]['cer_restored']:.3f}", flush=True)

    ef = tot["flag_wrong"] / max(tot["flag_n"], 1)
    ec = tot["clear_wrong"] / max(tot["clear_n"], 1)
    summary = {
        "mode": mode, "weights": args.weights, "n_images": len(rows), "ref_chars": tot["ref"],
        "cer_faded": tot["faded"] / max(tot["ref"], 1),
        "cer_clahe": tot["clahe"] / max(tot["ref"], 1),
        "cer_restored": tot["restored"] / max(tot["ref"], 1),
        "flagged_share": tot["flag_n"] / max(tot["flag_n"] + tot["clear_n"], 1),
        "error_rate_flagged": ef, "error_rate_unflagged": ec,
        "flag_error_ratio": (ef / ec) if ec > 0 else None,
        # Of all wrong characters in the reading, the share each signal flags.
        "errors_caught": tot["flag_wrong"] / max(tot["flag_wrong"] + tot["clear_wrong"], 1),
        "errors_caught_ensemble_only": tot["ens_wrong"] / max(tot["flag_wrong"] + tot["clear_wrong"], 1),
        "errors_caught_confidence_only": tot["conf_wrong"] / max(tot["flag_wrong"] + tot["clear_wrong"], 1),
        "flagged_share_ensemble_only": tot["ens_n"] / max(tot["flag_n"] + tot["clear_n"], 1),
        "flagged_share_confidence_only": tot["conf_n"] / max(tot["flag_n"] + tot["clear_n"], 1),
        "error_rate_ensemble_flags": tot["ens_wrong"] / max(tot["ens_n"], 1),
        "error_rate_confidence_flags": tot["conf_wrong"] / max(tot["conf_n"], 1),
        "seconds_per_image": (time.time() - t0) / len(rows),
        "per_image": rows,
    }
    out = Path("outputs/eval")
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.name or ('synthetic' if args.clean else 'real')}.json"
    dest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))
    print("saved", dest)


if __name__ == "__main__":
    main()
