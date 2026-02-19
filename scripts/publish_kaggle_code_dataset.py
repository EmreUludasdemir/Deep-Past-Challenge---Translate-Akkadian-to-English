import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_FILES = [
    "kaggle_submit.py",
    "train.py",
    "inference.py",
    "utils.py",
    "requirements.txt",
    "README.md",
    "configs/baseline.yaml",
    "configs/top_lb.yaml",
]


def _has_kaggle_auth() -> bool:
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    return cfg.exists()


def _copy_payload(repo_root: Path, payload_root: Path, files: list[str]) -> None:
    dataset_folder = payload_root / "deep-past-code-dataset"
    dataset_folder.mkdir(parents=True, exist_ok=True)
    for rel in files:
        src = repo_root / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")
        dst = dataset_folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_metadata(payload_root: Path, dataset_id: str) -> None:
    title = dataset_id.split("/")[-1]
    metadata = {
        "id": dataset_id,
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(payload_root / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the current code snapshot as a new Kaggle Dataset version."
    )
    parser.add_argument(
        "--dataset-id",
        default="emreuludasdemir/deep-past-codev1",
        help="Target Kaggle dataset id in <owner>/<slug> format.",
    )
    parser.add_argument(
        "--message",
        default="Update code dataset from local workspace",
        help="Version message shown on Kaggle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build payload and print command without publishing.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    with tempfile.TemporaryDirectory(prefix="kaggle_code_publish_") as tmp:
        payload_root = Path(tmp)
        _copy_payload(repo_root, payload_root, DEFAULT_FILES)
        _write_metadata(payload_root, args.dataset_id)

        cmd = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(payload_root),
            "-m",
            args.message,
            "-r",
            "zip",
        ]
        print("Publishing command:", " ".join(cmd))
        if args.dry_run:
            print("Dry-run enabled; no upload performed.")
            return
        if not _has_kaggle_auth():
            raise RuntimeError(
                "Kaggle auth not found. Set KAGGLE_API_TOKEN (recommended) or "
                "KAGGLE_USERNAME/KAGGLE_KEY, or configure ~/.kaggle/kaggle.json."
            )
        subprocess.run(cmd, check=True)
        print("Dataset version published:", args.dataset_id)


if __name__ == "__main__":
    main()
