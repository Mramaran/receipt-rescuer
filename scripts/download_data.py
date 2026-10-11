"""Download the clean receipt scans used as training ground truth.

Source: Voxel51/scanned_receipts on Hugging Face (the SROIE receipt scans,
CC-BY-4.0). Nothing from this dataset is committed to the repository.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Voxel51/scanned_receipts")
    ap.add_argument("--out", default="data/sroie")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(out))
    n = sum(1 for _ in out.rglob("*.jpg"))
    print(f"downloaded {n} images to {out}")


if __name__ == "__main__":
    main()
