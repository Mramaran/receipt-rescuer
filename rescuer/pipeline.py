"""End-to-end: faded image -> restored image, reading, flags, overlay."""
import cv2
import numpy as np

from .flag import CONF_THRESH, flag_chars
from .infer import load_model, restore
from .ocr import get_backend


class Rescuer:
    def __init__(self, weights, ocr="tesseract", device=None):
        self.model, self.device, self.info = load_model(weights, device)
        self.ocr, self.ocr_error = None, None
        try:
            self.ocr = get_backend(ocr)
        except Exception as e:  # the app still restores images without OCR
            self.ocr_error = str(e)

    def run(self, img, conf_thresh=CONF_THRESH):
        """img: float32 gray in [0, 1]. Returns a dict; reading/flags are None without OCR."""
        res = restore(img, self.model, self.device)
        out = {"restored": res["mean"], "std": res["std"], "reading": None, "flags": None,
               "member_texts": None}
        if self.ocr is not None:
            reading = self.ocr.read(res["mean"])
            member_texts = [self.ocr.read(m).text for m in res["members"]]
            out.update(reading=reading, member_texts=member_texts,
                       flags=flag_chars(reading, member_texts, conf_thresh=conf_thresh))
        return out


def render_overlay(restored, std, reading=None, flags=None):
    """Restored image with ensemble disagreement as an amber tint and flagged characters boxed."""
    base = cv2.cvtColor(np.clip(restored * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    scale = max(float(np.percentile(std, 99.5)), 0.02)
    heat = cv2.GaussianBlur(np.clip(std / scale, 0, 1), (0, 0), 2.0)
    alpha = (0.75 * heat)[..., None]
    amber = np.array([0, 150, 255], dtype=np.float32)  # BGR
    img = (base.astype(np.float32) * (1 - alpha) + amber * alpha).astype(np.uint8)
    if reading is not None and flags is not None:
        for ch, f in zip(reading.chars, flags):
            if f["flagged"]:
                x0, y0, x1, y1 = ch.box
                cv2.rectangle(img, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (40, 40, 220), 1)
    return img
