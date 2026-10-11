# Receipt Rescuer

Restore a faded thermal receipt, read it, and mark every character the system is unsure about.

Built for Banana Hacks 2026 (Image AI).

## The problem

Thermal receipts fade until the amount, date and invoice number can't be read. Scanner apps brighten the photo, but a restored "8" that was really a "3" looks just as confident as a correct one. Receipt Rescuer returns the text with the doubtful characters highlighted, so you know which digits to check against the paper.

## What it does

1. **Restores** the photo with a small network trained on synthetically faded receipts.
2. **Reads** the restored image with an OCR engine that reports a confidence per character.
3. **Flags** a character when either signal fires:
   - the receipt was restored four times under different flips, and the readings disagree at that position;
   - the OCR confidence is low.
4. **Shows** the flagged text, the restored image, and an overlay of where the restoration passes disagreed.

A highlight means "check this". It does not mean the character is wrong, and no highlight does not prove a character is right.

## Results

Measured on 20 receipt scans held out from training (13,419 reference characters), faded synthetically at three strengths with a fixed seed. Weights: `weights/rescuenet.pt`, the best checkpoint of a 25,000-step run.

| Fade strength | Faded input CER | CLAHE baseline CER | Restored CER | Share flagged | Error rate, flagged | Error rate, unflagged | Errors caught by flags |
|---|---|---|---|---|---|---|---|
| 0.4 | 0.62 | 0.77 | **0.24** | 45% | 42% | 6.5% | 84% |
| 0.6 | 0.68 | 0.84 | **0.32** | 50% | 46% | 8.4% | 85% |
| 0.8 | 0.83 | 0.99 | **0.47** | 65% | 57% | 15% | 88% |
CER is character error rate (lower is better). A flagged character is 4 to 6.5 times as likely to be wrong as an unflagged one, and the flags cover 84% to 88% of all wrong characters.

The app flags whole words: when one character of a word is doubtful, the rest of the word is highlighted too, in a paler colour. That is why the flagged share is high. The same receipts scored with character-level flags only (`flag_chars(..., whole_words=False)`):

| Fade strength | Share flagged | Error rate, flagged | Error rate, unflagged | Errors caught |
|---|---|---|---|---|
| 0.4 | 22% | 69% | 9% | 68% |
| 0.6 | 26% | 71% | 12% | 68% |
| 0.8 | 36% | 78% | 22% | 67% |

What each signal contributes at character level (fade strength 0.6):

| Signal | Share of characters flagged | Wrong characters caught |
|---|---|---|
| Restoration passes disagree | 21% | 58% |
| OCR confidence below 95 | 19% | 51% |
| Either | 26% | 68% |
| Either, spread to whole words (what the app uses) | 50% | 85% |

How to read these numbers:

- The reference text is the OCR reading of the clean scan, so they measure how much of what OCR can read on a clean receipt is recovered after fading. Tesseract itself makes some mistakes on the clean scans.
- The fading is synthetic and comes from the same generator the network was trained on (on different receipts). Real faded receipts may behave differently; they have not been measured.
- Unflagged characters are still wrong 6.5% to 15% of the time, and roughly one wrong character in seven carries no flag. The flags narrow down where to look; they do not certify the rest.
- Some scans are damaged before any fading (a misprinted price, for example). Restoration reproduces that damage; it does not fill in what was never printed.
- The OCR-confidence threshold was chosen on 12 other held-out scans, which are excluded from this table (`--skip 20`).

Reproduce with `python evaluate.py --weights weights/rescuenet.pt --clean data/sroie --skip 20 --n 20 --severity 0.6`.

## Setup

Requires Python 3.10+, a CUDA GPU for training, and the Tesseract OCR engine.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Install Tesseract separately (on Windows: `winget install UB-Mannheim.TesseractOCR`). If it is not on `PATH`, set `TESSERACT_CMD` to the full path of `tesseract.exe`.

## Run

```bash
# 1. Clean receipt scans used as training ground truth (about 740 MB)
python scripts/download_data.py

# 2. Preview the synthetic fading at three strengths
python scripts/preview_degrade.py --n 6

# 3. Train (settings, seed and log are written to runs/<name>/)
python train.py --data data/sroie --name run1 --iters 30000

# 4. Measure
python evaluate.py --weights runs/run1/best.pt --clean data/sroie --n 40 --severity 0.8
python evaluate.py --weights runs/run1/best.pt --real data/real

# 5. Web app at http://localhost:8000
python -m uvicorn app.server:app --port 8000
```

For real-receipt evaluation, put photos in `data/real/` with a hand transcript `<name>.txt` beside each one.

Fast checks that need no GPU, data or OCR engine: `python tests/test_core.py`.

## How it works

| Part | File | Notes |
|---|---|---|
| Synthetic fading | `rescuer/degrade.py` | Patchy ink loss, faint strokes vanish first, print-head streaks, paper tone drift, then phone-capture shadow, blur, noise and JPEG |
| Network | `rescuer/model.py` | NAFNet-style encoder/decoder, grayscale, width 32 |
| Training | `train.py` | Random patches faded on the fly, Charbonnier loss, mixed precision, EMA weights |
| Ensemble inference | `rescuer/infer.py` | Four flip variants; the mean is the output, the spread is the uncertainty |
| OCR | `rescuer/ocr.py` | Tesseract hOCR with per-character confidence, behind a small interface |
| Flagging | `rescuer/flag.py` | Aligns each ensemble member's reading to the final reading |
| Evaluation | `evaluate.py`, `rescuer/metrics.py` | CER for faded / CLAHE / restored, and error rate of flagged versus unflagged characters |
| Web app | `app/` | FastAPI server and a single static page |

## Limits

- Printed English thermal receipts only. No handwriting, no other document types.
- No perspective correction: crop the photo to the receipt first.
- Where no ink is left on the paper, nothing can be recovered; text that appears there is a guess.
- The flags are a prompt to check, not a verification. The measured error rates are in the Results table.
- Not for proving a receipt is authentic.

## Third-party components

| Component | Use | Licence |
|---|---|---|
| [SROIE receipt scans via Voxel51/scanned_receipts](https://huggingface.co/datasets/Voxel51/scanned_receipts) | Clean training and evaluation images (not redistributed here) | CC-BY-4.0 |
| NAFNet architecture (Chen et al., ECCV 2022) | Design of the network; implementation written for this project | Paper |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Text reading with per-character confidence | Apache-2.0 |
| PyTorch, OpenCV, NumPy, FastAPI, Uvicorn, huggingface_hub | Libraries | BSD / Apache-2.0 / MIT |
