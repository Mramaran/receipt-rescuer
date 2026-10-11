"""OCR with per-character confidence, behind a small interface.

The Tesseract backend asks for hOCR output with character boxes, which carries
a confidence value for every character.
"""
import html
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Char:
    text: str
    conf: float          # 0-100
    box: tuple           # x0, y0, x1, y1 in image pixels
    pos: int             # index of this character in OcrResult.text


@dataclass
class OcrResult:
    text: str = ""
    chars: list = field(default_factory=list)


_TOKEN = re.compile(
    r"<span class='(?P<line>ocr_line|ocr_caption|ocr_textfloat|ocr_header)'"
    r"|<span class='(?P<word>ocrx_word)'"
    r"|<span class='ocrx_cinfo' title='x_bboxes (?P<x0>\d+) (?P<y0>\d+) (?P<x1>\d+) (?P<y1>\d+);"
    r" x_conf (?P<conf>[\d.]+)'>(?P<ch>.*?)</span>",
    re.S,
)


def parse_hocr(hocr):
    """Turn Tesseract hOCR (with hocr_char_boxes=1) into text plus per-character data."""
    parts, chars, n = [], [], 0
    for m in _TOKEN.finditer(hocr):
        if m.group("line"):
            if n and parts[-1] != "\n":
                parts.append("\n")
                n += 1
        elif m.group("word"):
            if n and parts[-1] not in ("\n", " "):
                parts.append(" ")
                n += 1
        else:
            ch = html.unescape(m.group("ch"))
            if not ch:
                continue
            box = tuple(int(m.group(k)) for k in ("x0", "y0", "x1", "y1"))
            chars.append(Char(ch, float(m.group("conf")), box, n))
            parts.append(ch)
            n += len(ch)
    return OcrResult("".join(parts), chars)


def find_tesseract():
    """Path to the tesseract executable, or None."""
    try:
        on_path = shutil.which("tesseract")
    except OSError:
        on_path = None
    candidates = [os.environ.get("TESSERACT_CMD"), on_path,
                  r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                  str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe")]
    for c in candidates:
        try:
            if c and Path(c).is_file():
                return c
        except OSError:  # unreadable location; treat as not found
            continue
    return None


class TesseractBackend:
    name = "tesseract"

    def __init__(self, psm=6, lang="eng"):
        cmd = find_tesseract()
        if cmd is None:
            raise RuntimeError("Tesseract is not installed. Install it and reopen the terminal, "
                               "or set TESSERACT_CMD to the path of tesseract.exe.")
        # The image goes in on stdin and hOCR comes back on stdout: no temp files.
        self._args = [cmd, "stdin", "stdout", "--psm", str(psm), "-l", lang,
                      "-c", "hocr_char_boxes=1", "hocr"]
        self._flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def read(self, img):
        """img: float32 gray in [0, 1]. Returns OcrResult."""
        ok, png = cv2.imencode(".png", np.clip(img * 255, 0, 255).astype(np.uint8))
        proc = subprocess.run(self._args, input=png.tobytes(), capture_output=True,
                              creationflags=self._flags, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError("tesseract failed: " + proc.stderr.decode("utf-8", "replace")[:300])
        return parse_hocr(proc.stdout.decode("utf-8", errors="replace"))


def get_backend(name="tesseract", **kw):
    if name == "tesseract":
        return TesseractBackend(**kw)
    raise ValueError(f"unknown OCR backend: {name}")
