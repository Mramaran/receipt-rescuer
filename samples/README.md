# Sample receipts

These four images are what the "Try a sample" buttons in the web app load.

- They are **synthetically faded**. Each is a scan from the SROIE receipt set (via [Voxel51/scanned_receipts](https://huggingface.co/datasets/Voxel51/scanned_receipts), CC-BY-4.0) that was held out from training, faded by `rescuer/degrade.py` with the same severity and seed used in the evaluation.
- Samples 1 to 3 were picked because they restore well. Sample 4 was picked because almost nothing can be recovered from it. The average behaviour is in the Results table of the main README, not in these four.
- Regenerate them with `python scripts/make_samples.py`. `samples.json` records the source scan and fade severity of each.
