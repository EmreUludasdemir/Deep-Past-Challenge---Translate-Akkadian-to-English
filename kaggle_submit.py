"""
Kaggle notebook wrapper for Deep Past MT pipeline.

Usage in a Kaggle notebook cell:
    !python kaggle_submit.py --track baseline
    !python kaggle_submit.py --track top_lb
    !python kaggle_submit.py --track baseline --inference-only --weights-dir /kaggle/input/akkadian-weights
    !python kaggle_submit.py --track top_lb --inference-only --weights-dir /kaggle/input/akkadian-weights

Modes:
    Full run (default):  train + inference within GPU time limit.
    Inference-only:      load pre-trained weights from a Kaggle Dataset, run inference only.
"""

import argparse
import json
import os
import shutil
import time
from typing import Any, Dict, List, Tuple

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

HF_MODEL_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
)

HF_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer_config.json",
)


def resolve_config_path(track: str) -> str:
    rel = TRACK_CONFIGS.get(track)
    if rel is None:
        raise ValueError(f"Unknown track '{track}'. Choose from: {list(TRACK_CONFIGS.keys())}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, rel)


def _is_hf_checkpoint_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_model = any(os.path.exists(os.path.join(path, f)) for f in HF_MODEL_FILENAMES)
    has_tokenizer = any(os.path.exists(os.path.join(path, f)) for f in HF_TOKENIZER_FILENAMES)
    return has_config and has_model and has_tokenizer


def _bind_checkpoints_from_weights_dir(
    config: Dict[str, Any],
    weights_dir: str,
) -> Tuple[List[str], List[str]]:
    bound_models: List[str] = []
    issues: List[str] = []

    for model_cfg in config["models"]:
        model_name = str(model_cfg["name"])
        ckpt_dir = os.path.join(weights_dir, model_name)
        if not os.path.isdir(ckpt_dir):
            issues.append(f"missing directory: {ckpt_dir}")
            continue
        if not _is_hf_checkpoint_dir(ckpt_dir):
            issues.append(
                "invalid Hugging Face checkpoint layout: "
                f"{ckpt_dir} (need config + model + tokenizer files)"
            )
            continue
        model_cfg["checkpoint"] = ckpt_dir
        bound_models.append(model_name)
    return bound_models, issues


def _validate_summary_checkpoints(summary_path: str) -> List[str]:
    missing: List[str] = []
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for model in summary.get("models", []):
        stages = model.get("stages", [])
        if not stages:
            continue
        best_stage = sorted(stages, key=lambda x: x.get("val_geo_mean", -1.0))[-1]
        checkpoint = str(best_stage.get("checkpoint_dir", "")).strip()
        if not checkpoint or not os.path.isdir(checkpoint):
            model_name = str(model.get("name", "unknown_model"))
            missing.append(f"{model_name}: {checkpoint or '<empty checkpoint_dir>'}")
    return missing


def _assert_competition_data_dir(data_dir: str) -> None:
    required_files = ("train.csv", "test.csv", "sample_submission.csv")
    missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            "Competition dataset is missing required files under data_dir.\n"
            f"data_dir: {data_dir}\n"
            f"missing: {missing}"
        )


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
        "--use-weights-summary",
        action="store_true",
        help=(
            "Use --weights-dir/training_summary.json to resolve checkpoints. "
            "Disabled by default because stale absolute paths are common in offline Kaggle runs."
        ),
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
    _assert_competition_data_dir(config["global"]["data_dir"])

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
            if not os.path.isdir(args.weights_dir):
                raise FileNotFoundError(f"--weights-dir does not exist: {args.weights_dir}")
            print(f"=== INFERENCE-ONLY MODE (weights from {args.weights_dir}) ===")
            summary_candidate = os.path.join(args.weights_dir, "training_summary.json")
            if args.use_weights_summary:
                if not os.path.exists(summary_candidate):
                    raise FileNotFoundError(
                        f"--use-weights-summary was set, but file was not found: {summary_candidate}"
                    )
                missing = _validate_summary_checkpoints(summary_candidate)
                if missing:
                    missing_lines = "\n  - " + "\n  - ".join(missing)
                    raise FileNotFoundError(
                        "training_summary.json points to missing checkpoint directories:"
                        f"{missing_lines}\n"
                        "Use direct model folders under --weights-dir, or fix summary paths."
                    )
                summary_path = summary_candidate
                print(f"Using summary: {summary_path}")
            else:
                if os.path.exists(summary_candidate):
                    print(
                        "Found training_summary.json in --weights-dir, but ignoring it by default. "
                        "Pass --use-weights-summary to force summary-based resolution."
                    )
                bound_models, issues = _bind_checkpoints_from_weights_dir(config, args.weights_dir)
                summary_path = ""
                expected_names = [str(m["name"]) for m in config["models"]]
                missing_models = [name for name in expected_names if name not in bound_models]
                if missing_models:
                    expected_layout = "\n  - " + "\n  - ".join(
                        [os.path.join(args.weights_dir, name) for name in expected_names]
                    )
                    issue_block = ""
                    if issues:
                        issue_block = "\nIssues:\n  - " + "\n  - ".join(issues)
                    raise FileNotFoundError(
                        "Could not resolve all model checkpoints from --weights-dir.\n"
                        f"Expected model directories:{expected_layout}{issue_block}"
                    )
                print(
                    f"Resolved direct checkpoints for {len(bound_models)}/"
                    f"{len(expected_names)} models."
                )
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
