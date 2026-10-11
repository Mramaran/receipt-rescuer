"""Web app: upload a faded receipt, get flagged text, the restored image and an overlay.

    python -m uvicorn app.server:app --port 8000

Weights are taken from RESCUER_WEIGHTS, else weights/rescuenet.pt, else the
newest runs/*/best.pt.
"""
import base64
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from rescuer.degrade import thermal_fade
from rescuer.infer import prepare
from rescuer.pipeline import Rescuer, render_overlay

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Receipt Rescuer")
_lock = threading.Lock()
_rescuer = None
_load_error = None


def _find_weights():
    env = os.environ.get("RESCUER_WEIGHTS")
    if env:
        return Path(env)
    fixed = ROOT / "weights" / "rescuenet.pt"
    if fixed.exists():
        return fixed
    runs = sorted((ROOT / "runs").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


_loaded = None  # (path, mtime) of the weights in memory


def _get():
    """Return the Rescuer, reloading when a newer weights file appears (e.g. during training)."""
    global _rescuer, _load_error, _loaded
    path = _find_weights()
    if path is None or not path.exists():
        if _rescuer is None:
            _load_error = "No trained weights found. Train a model first (see README)."
        return _rescuer
    key = (str(path), path.stat().st_mtime)
    if key != _loaded:
        try:
            _rescuer = Rescuer(path)
            _loaded, _load_error = key, None
        except Exception as e:  # a half-written file keeps the previous model in use
            if _rescuer is None:
                _load_error = f"Could not load weights: {e}"
    return _rescuer


def _png(img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def _lines(reading, flags):
    """Group characters into lines for display, keeping spaces between words."""
    by_pos = {f["pos"]: (f, ch) for f, ch in zip(flags, reading.chars)}
    lines, cur, i = [], [], 0
    text = reading.text
    while i < len(text):
        if text[i] == "\n":
            lines.append(cur)
            cur = []
            i += 1
        elif i in by_pos:
            f, ch = by_pos[i]
            item = {"c": f["char"], "conf": round(f["conf"]), "dis": round(f["disagree"], 2),
                    "flag": f["flagged"], "why": f["why"]}
            if f["flagged"]:
                item["box"] = list(ch.box)
            cur.append(item)
            i += len(f["char"])
        else:
            cur.append({"c": text[i]})
            i += 1
    lines.append(cur)
    return [t for t in map(_tidy, lines) if t]


def _tidy(line):
    """Remove OCR clutter that is not receipt text. Display only; the evaluation uses raw text.

    Drops lines with no letter or digit (specks and edge shadows read as punctuation),
    stray '|' marks from the paper edge at the start of a line, and the gap the
    reader inserts after a decimal point ("RM2. 70" -> "RM2.70").
    """
    if not any(it["c"].isalnum() for it in line):
        return []
    while line and line[0]["c"] in "| ":
        line = line[1:]
    out = []
    for i, it in enumerate(line):
        gap_after_point = (it["c"] == " " and len(out) >= 2 and out[-1]["c"] == "." and out[-2]["c"].isdigit()
                           and i + 1 < len(line) and line[i + 1]["c"].isdigit())
        if not gap_after_point:
            out.append(it)
    return out


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/health")
def health():
    r = _get()
    return {"model": r is not None, "model_error": _load_error,
            "weights": r.info if r else None,
            "ocr": bool(r and r.ocr), "ocr_error": r.ocr_error if r else None}


def _samples():
    meta = ROOT / "samples" / "samples.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else []


@app.get("/api/samples")
def samples():
    return [{"file": s["file"], "label": s["label"]} for s in _samples()]


@app.get("/samples/{name}")
def sample(name: str):
    if name not in {s["file"] for s in _samples()}:  # only files listed in samples.json
        raise HTTPException(404, "No such sample.")
    return FileResponse(ROOT / "samples" / name)


@app.post("/api/rescue")
def rescue(file: UploadFile = File(...), simulate_fade: float = Form(0.0)):
    r = _get()
    if r is None:
        raise HTTPException(503, _load_error)
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB.")
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "That file is not an image this app can read.")

    gray = prepare(img)
    simulated = 0.0 < simulate_fade <= 1.0
    if simulated:  # testing aid: fade a clean receipt before restoring it
        gray = thermal_fade(gray, np.random.default_rng(0), simulate_fade)

    t0 = time.time()
    with _lock:
        res = r.run(gray)
        before = r.ocr.read(gray).text if r.ocr else None
    reading, flags = res["reading"], res["flags"]
    stats = {"seconds": round(time.time() - t0, 1), "simulated_fade": simulated,
             "model_step": r.info.get("iter"),
             "width": int(gray.shape[1]), "height": int(gray.shape[0])}
    lines = _lines(reading, flags) if reading else None
    if lines is not None:
        shown = [it for line in lines for it in line if "conf" in it]
        n, nf = len(shown), sum(it["flag"] for it in shown)
        stats.update(chars=n, flagged=nf, flagged_share=round(nf / max(n, 1), 3))
    return {
        "lines": lines,
        "before_text": before,
        "ocr_error": r.ocr_error,
        "stats": stats,
        "images": {"input": _png(gray), "restored": _png(res["restored"]),
                   "overlay": _png(render_overlay(res["restored"], res["std"], reading, flags))},
    }
