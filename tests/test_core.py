"""Fast checks that need no GPU, no data and no OCR engine.

    python -m pytest tests -q      (or: python tests/test_core.py)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rescuer.degrade import clahe_baseline, thermal_fade  # noqa: E402
from rescuer.flag import agreement, flag_chars  # noqa: E402
from rescuer.infer import restore  # noqa: E402
from rescuer.metrics import cer, flag_counts, levenshtein  # noqa: E402
from rescuer.model import RescueNet  # noqa: E402
from rescuer.ocr import parse_hocr  # noqa: E402

HOCR = (
    "<span class='ocr_line' id='line_1_1' title='bbox 0 0 90 20'>"
    "<span class='ocrx_word' id='word_1_1' title='bbox 0 0 40 20; x_wconf 90'>"
    "<span class='ocrx_cinfo' title='x_bboxes 0 0 10 20; x_conf 99.1'>T</span>"
    "<span class='ocrx_cinfo' title='x_bboxes 10 0 20 20; x_conf 42.5'>&amp;</span></span> "
    "<span class='ocrx_word' id='word_1_2' title='bbox 50 0 90 20; x_wconf 80'>"
    "<span class='ocrx_cinfo' title='x_bboxes 50 0 60 20; x_conf 88'>9</span></span></span>"
    "<span class='ocr_line' id='line_1_2' title='bbox 0 30 90 50'>"
    "<span class='ocrx_word' id='word_1_3' title='bbox 0 30 10 50; x_wconf 70'>"
    "<span class='ocrx_cinfo' title='x_bboxes 0 30 10 50; x_conf 70'>x</span></span></span>"
)


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("flaw", "lawn") == 2


def test_cer_ignores_whitespace_layout():
    assert cer("TOTAL  12.50\n", "TOTAL 12.50") == (0, 11)
    assert cer("12.50", "12.80") == (1, 5)


def test_parse_hocr():
    r = parse_hocr(HOCR)
    assert r.text == "T& 9\nx"
    assert [c.text for c in r.chars] == ["T", "&", "9", "x"]
    assert [c.pos for c in r.chars] == [0, 1, 3, 5]
    assert r.chars[1].conf == 42.5 and r.chars[2].box == (50, 0, 60, 20)


def test_agreement_and_flags():
    assert agreement("12.50", "12.80").tolist() == [True, True, True, False, True]
    ref = parse_hocr(HOCR)                       # "T& 9\nx", '&' has low confidence
    members = ["T& 9\nx", "T& 8\nx", "T& 8\nx", "T& 9\nx"]
    spread = flag_chars(ref, members, conf_thresh=60)      # '&' is doubtful, so all of "T&" is flagged
    assert [f["why"] for f in spread] == ["word", "confidence", "ensemble", ""]
    flags = flag_chars(ref, members, conf_thresh=60, whole_words=False)
    assert [f["flagged"] for f in flags] == [False, True, True, False]
    assert [f["why"] for f in flags] == ["", "confidence", "ensemble", ""]
    counts = flag_counts(ref.text, flags, "T& 8\nx")
    assert counts == {"flag_n": 2, "flag_wrong": 1, "clear_n": 2, "clear_wrong": 0,
                      "ens_n": 1, "ens_wrong": 1, "conf_n": 1, "conf_wrong": 0}


def test_fade_reduces_contrast():
    rng = np.random.default_rng(0)
    clean = np.ones((160, 200), np.float32)
    clean[40:120:8, 20:180] = 0.05               # dark text-like stripes
    faded = thermal_fade(clean, rng, severity=0.9)
    assert faded.shape == clean.shape and faded.dtype == np.float32
    assert 0.0 <= faded.min() and faded.max() <= 1.0
    assert faded.std() < 0.5 * clean.std()
    assert clahe_baseline(faded).shape == clean.shape


def test_model_and_ensemble_shapes():
    model = RescueNet(width=8, enc_blocks=(1, 1), mid_blocks=1, dec_blocks=(1, 1)).eval()
    x = torch.rand(1, 1, 37, 53)                 # not a multiple of the stride
    assert model(x).shape == x.shape
    out = restore(np.random.rand(37, 53).astype(np.float32), model, torch.device("cpu"))
    assert out["mean"].shape == (37, 53) and len(out["members"]) == 4
    assert out["std"].shape == (37, 53)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
