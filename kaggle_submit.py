"""
Kaggle notebook wrapper for Deep Past MT pipeline.

Usage in a Kaggle notebook cell:
    !python kaggle_submit.py --track baseline
    !python kaggle_submit.py --track top_lb
    !python kaggle_submit.py --track top_lb --inference-only --weights-dir /kaggle/input/akkadian-weights

Modes:
    Full run (default):  train + inference within GPU time limit.
    Inference-only:      load pre-trained weights from a Kaggle Dataset, run inference only.
"""

import argparse
import os
import shutil
import sys
import time

IS_KAGGLE = os.path.exists("/kaggle/input")

DEFAULT_DATA_DIR = (
    "/kaggle/input/deep-past-initiative-machine-translation"
    if IS_KAGGLE
    else os.path.join(os.path.dirname(__file__), "data")
)
DEFAULT_OUTPUT_DIR = "/kaggle/working" if IS_KAGGLE else os.path.join(os.path.dirname(__file__), "outputs")

TRACK_CONFIGS = {
    "baseline": "configs/baseline.yaml",
    "top_lb": "configs/top_lb.yaml",
}


def resolve_config_path(track: str) -> str:
    rel = TRACK_CONFIGS.get(track)
    if rel is None:
        raise ValueError(f"Unknown track '{track}'. Choose from: {list(TRACK_CONFIGS.keys())}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, rel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle submission wrapper")
    parser.add_argument(
        "--track",
        type=str,
        default="baseline",
        choices=list(TRACK_CONFIGS.keys()),
        help="Which config track to use",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip training; load weights from --weights-dir",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="",
        help="Directory with pre-trained model weights (for inference-only mode)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="50-row smoke run for quick validation",
    )
    args = parser.parse_args()

    from utils import load_yaml

    config_path = resolve_config_path(args.track)
    config = load_yaml(config_path)

    config["global"]["data_dir"] = DEFAULT_DATA_DIR
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.track)
    config["global"]["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.smoke:
        config.setdefault("debug", {})["smoke_max_rows"] = 50

    summary_path = os.path.join(output_dir, "training_summary.json")

    if not args.inference_only:
        print(f"=== TRAINING ({args.track}) ===")
        t0 = time.time()

        from train import run_training

        summary = run_training(config)
        elapsed = time.time() - t0
        print(f"Training complete in {elapsed:.0f}s")
        print(f"Summary: {summary_path}")
    else:
        if args.weights_dir:
            print(f"=== INFERENCE-ONLY MODE (weights from {args.weights_dir}) ===")
            summary_candidate = os.path.join(args.weights_dir, "training_summary.json")
            if os.path.exists(summary_candidate):
                summary_path = summary_candidate
            else:
                for model_cfg in config["models"]:
                    ckpt_dir = os.path.join(args.weights_dir, model_cfg["name"])
                    if os.path.exists(ckpt_dir):
                        model_cfg["checkpoint"] = ckpt_dir
                summary_path = ""
        else:
            print("=== INFERENCE-ONLY MODE (using existing summary) ===")

    print(f"=== INFERENCE ({args.track}) ===")
    t0 = time.time()

    from inference import run_inference

    submission_path = run_inference(config, summary_path=summary_path)
    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.0f}s")
    print(f"Submission: {submission_path}")

    final_dest = os.path.join(DEFAULT_OUTPUT_DIR, "submission.csv")
    if submission_path != final_dest:
        shutil.copy2(submission_path, final_dest)
        print(f"Copied to: {final_dest}")


if __name__ == "__main__":
    main()
