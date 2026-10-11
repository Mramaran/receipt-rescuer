"""End-to-end check of the demo path through the web server.

Needs trained weights, Tesseract and the bundled samples. Takes about a minute.

    python tests/test_demo_path.py
"""
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import app.server as server  # noqa: E402

client = TestClient(server.app)


def _post(path, **form):
    with open(path, "rb") as f:
        return client.post("/api/rescue", files={"file": (Path(path).name, f, "image/png")}, data=form)


def test_page_and_health():
    assert client.get("/").status_code == 200
    h = client.get("/api/health").json()
    assert h["model"], h["model_error"]
    assert h["ocr"], h["ocr_error"]


def test_samples_are_served():
    samples = client.get("/api/samples").json()
    assert len(samples) >= 3
    for s in samples:
        r = client.get("/samples/" + s["file"])
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get("/samples/..%2Fweights%2Frescuenet.pt").status_code == 404
    assert client.get("/samples/nope.png").status_code == 404


def test_rescue_a_sample():
    r = _post(ROOT / "samples" / "sample1.png")
    assert r.status_code == 200
    d = r.json()
    chars = [c for line in d["lines"] for c in line]
    flagged = [c for c in chars if c.get("flag")]
    assert d["stats"]["chars"] > 200, "the sample should restore to readable text"
    assert 0 < len(flagged) < 0.5 * d["stats"]["chars"]
    assert d["stats"]["flagged"] == len(flagged)
    w, h = d["stats"]["width"], d["stats"]["height"]
    for c in flagged:  # every flagged character can be located on the photo
        x0, y0, x1, y1 = c["box"]
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert c["why"] in ("ensemble", "confidence", "both", "word")
    for k in ("input", "restored", "overlay"):
        head, b64 = d["images"][k].split(",", 1)
        assert head == "data:image/png;base64" and base64.b64decode(b64)[:4] == b"\x89PNG"


def test_unrecoverable_sample_does_not_crash():
    r = _post(ROOT / "samples" / "sample4.png")
    assert r.status_code == 200
    assert r.json()["lines"] is not None


def test_test_mode_fades_first():
    r = _post(ROOT / "samples" / "sample1.png", simulate_fade="0.5")
    assert r.status_code == 200 and r.json()["stats"]["simulated_fade"] is True


def test_rejects_a_file_that_is_not_an_image():
    r = client.post("/api/rescue", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "not an image" in r.json()["detail"]


def test_missing_weights_gives_a_clear_error():
    saved = (server._rescuer, server._loaded, server._load_error, server._find_weights)
    try:
        server._rescuer, server._loaded, server._load_error = None, None, None
        server._find_weights = lambda: None
        assert client.get("/api/health").json()["model"] is False
        r = _post(ROOT / "samples" / "sample1.png")
        assert r.status_code == 503 and "weights" in r.json()["detail"]
    finally:
        server._rescuer, server._loaded, server._load_error, server._find_weights = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
