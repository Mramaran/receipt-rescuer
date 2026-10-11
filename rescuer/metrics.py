"""Character error rate, and how well the flags predict errors."""
import re

import numpy as np

from .flag import agreement


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a, b):
    """Edit distance between two strings (row-wise dynamic programme in numpy)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    b_codes = np.array([ord(c) for c in b], dtype=np.int64)
    idx = np.arange(len(b) + 1, dtype=np.int64)
    prev = idx.copy()
    for i, ch in enumerate(a, start=1):
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(prev[:-1] + (b_codes != ord(ch)), prev[1:] + 1)
        prev = np.minimum.accumulate(cur - idx) + idx   # insertions
    return int(prev[-1])


def cer(reference, hypothesis):
    """Returns (edits, reference_length) on whitespace-normalized text."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    return levenshtein(ref, hyp), len(ref)


def flag_counts(hyp_text, flags, reference):
    """Count right/wrong characters among flagged and unflagged ones.

    A hypothesis character counts as wrong when it does not align to an equal
    character of the reference. Returns counts for flagged and unflagged ("clear")
    characters, and for each signal on its own: "ens" (ensemble disagreement) and
    "conf" (low OCR confidence). A character flagged by both counts in both.
    """
    correct = agreement(hyp_text, reference)
    out = {k: 0 for k in ("flag_n", "flag_wrong", "clear_n", "clear_wrong",
                          "ens_n", "ens_wrong", "conf_n", "conf_wrong")}
    for f in flags:
        wrong = int(not bool(correct[f["pos"]:f["pos"] + len(f["char"])].all()))
        key = "flag" if f["flagged"] else "clear"
        out[key + "_n"] += 1
        out[key + "_wrong"] += wrong
        for sig, names in (("ens", ("ensemble", "both")), ("conf", ("confidence", "both"))):
            if f["why"] in names:
                out[sig + "_n"] += 1
                out[sig + "_wrong"] += wrong
    return out
