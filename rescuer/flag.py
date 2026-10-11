"""Decide which characters of the restored reading should be flagged.

A character is flagged when either signal fires:

* ensemble disagreement: the same receipt restored under different flips was
  read differently at this position by the OCR engine;
* low OCR confidence on the restored image.

Neither signal proves a character is wrong, and an unflagged character is not
proven right. evaluate.py measures how much more often flagged characters are
wrong.
"""
from difflib import SequenceMatcher

import numpy as np


def agreement(ref_text, other_text):
    """Boolean array over ref_text: True where `other_text` has the same character aligned."""
    same = np.zeros(len(ref_text), dtype=bool)
    sm = SequenceMatcher(None, ref_text, other_text, autojunk=False)
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag == "equal":
            same[i1:i2] = True
    return same


# Tesseract's per-character confidence is compressed: on restored receipts it ran from
# about 72 to 99.6, and characters below 95 were wrong about 70% of the time. The
# threshold was set on 12 held-out receipts that are excluded from the reported results.
CONF_THRESH = 95.0


def flag_chars(ref, member_texts, conf_thresh=CONF_THRESH, max_disagree=0.34, whole_words=True):
    """ref: OcrResult of the restored image. member_texts: readings of each ensemble member.

    Returns one dict per character in ref.chars:
    pos, char, conf, disagree (0-1), flagged, why ("", "ensemble", "confidence", "both").
    """
    if member_texts:
        agree = np.mean([agreement(ref.text, t) for t in member_texts], axis=0)
    else:
        agree = np.ones(len(ref.text))
    out = []
    for ch in ref.chars:
        dis = float(1.0 - agree[ch.pos:ch.pos + len(ch.text)].min())
        by_ens = dis > max_disagree
        by_conf = ch.conf < conf_thresh
        why = "both" if by_ens and by_conf else "ensemble" if by_ens else "confidence" if by_conf else ""
        out.append({"pos": ch.pos, "char": ch.text, "conf": ch.conf, "disagree": dis,
                    "flagged": bool(by_ens or by_conf), "why": why})
    if whole_words:
        _spread_to_words(out)
    return out


def _spread_to_words(flags):
    """Flag every character of a word that contains a flagged character.

    A damaged word is usually misread in more than one place, and the reader
    is often confident about the wrong neighbours. Those get why="word".
    """
    start = 0
    for i in range(1, len(flags) + 1):
        end_of_word = i == len(flags) or flags[i]["pos"] != flags[i - 1]["pos"] + len(flags[i - 1]["char"])
        if end_of_word:
            word = flags[start:i]
            if any(f["flagged"] for f in word):
                for f in word:
                    if not f["flagged"]:
                        f["flagged"], f["why"] = True, "word"
            start = i
