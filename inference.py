import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import (
    build_named_entity_lexicon,
    canonicalize_text,
    consensus_rerank,
    load_competition_data,
    load_yaml,
    repair_named_entities,
    save_json,
    set_seed,
    validate_submission_df,
)


def _normalize_weights(raw_weights: Sequence[float]) -> List[float]:
    if not raw_weights:
        return []
    total = sum(max(w, 1e-9) for w in raw_weights)
    return [max(w, 1e-9) / total for w in raw_weights]


def _resolve_model_specs(
    config: Dict[str, Any], summary_path: str = ""
) -> List[Dict[str, Any]]:
    specs = []
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        for m in summary.get("models", []):
            stages = m.get("stages", [])
            if not stages:
                continue
            best_stage = sorted(stages, key=lambda x: x.get("val_geo_mean", -1.0))[-1]
            specs.append(
                {
                    "name": m["name"],
                    "checkpoint": best_stage["checkpoint_dir"],
                    "weight": max(float(best_stage.get("val_geo_mean", 1.0)), 1e-6),
                }
            )
        if specs:
            return specs

    for m in config["models"]:
        manual_ckpt = m.get("checkpoint")
        if manual_ckpt:
            specs.append({"name": m["name"], "checkpoint": manual_ckpt, "weight": 1.0})
    return specs


def _generate_candidates_for_model(
    model_name: str,
    checkpoint: str,
    sources: Sequence[str],
    gen_cfg: Dict[str, Any],
    batch_size: int,
) -> List[List[Dict[str, float]]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    n_best = int(gen_cfg.get("n_best", 1))
    num_beams = max(int(gen_cfg.get("num_beams", 4)), n_best)
    max_length = int(gen_cfg.get("max_target_length", 160))
    candidates: List[List[Dict[str, float]]] = []

    for start in range(0, len(sources), batch_size):
        batch = list(sources[start : start + batch_size])
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(gen_cfg.get("max_source_length", 256)),
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                num_beams=num_beams,
                num_return_sequences=n_best,
                max_length=max_length,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        seq_scores = out.sequences_scores.detach().cpu().tolist()

        for i in range(len(batch)):
            row = []
            for j in range(n_best):
                idx = i * n_best + j
                row.append(
                    {
                        "text": canonicalize_text(decoded[idx], is_translation=True),
                        "score": float(seq_scores[idx]),
                    }
                )
            candidates.append(row)

    print(f"[inference] {model_name}: generated {len(candidates)} rows from {checkpoint}")
    return candidates


def _weighted_vote(row_candidates: List[str], row_weights: List[float]) -> str:
    score_by_text: Dict[str, float] = defaultdict(float)
    for text, weight in zip(row_candidates, row_weights):
        score_by_text[text] += weight
    ranked = sorted(score_by_text.items(), key=lambda kv: (-kv[1], len(kv[0])))
    return ranked[0][0]


def run_inference(config: Dict[str, Any], summary_path: str = "") -> str:
    set_seed(int(config["global"]["seed"]))
    frames = load_competition_data(config["global"]["data_dir"])
    test_df = frames["test"].copy()
    sample_submission = frames.get("sample_submission")
    lexicon = build_named_entity_lexicon(frames.get("oa_lexicon"))

    source_col = config["global"].get("source_column", "transliteration")
    target_col = config["global"].get("target_column", "translation")
    id_col = config["global"].get("id_column", "id")

    test_df["source_norm"] = test_df[source_col].fillna("").astype(str).map(
        lambda x: canonicalize_text(x, is_translation=False)
    )
    sources = test_df["source_norm"].tolist()

    model_specs = _resolve_model_specs(config, summary_path=summary_path)
    if not model_specs:
        raise RuntimeError(
            "No checkpoints resolved. Train first or set model.checkpoint in config."
        )

    raw_weights = [float(spec.get("weight", 1.0)) for spec in model_specs]
    norm_weights = _normalize_weights(raw_weights)
    for spec, w in zip(model_specs, norm_weights):
        spec["norm_weight"] = w

    batch_size = int(config["inference"].get("batch_size", 8))
    n_best = int(config["inference"].get("n_best", 1))

    all_candidates: Dict[str, List[List[Dict[str, float]]]] = {}
    for spec in model_specs:
        all_candidates[spec["name"]] = _generate_candidates_for_model(
            model_name=spec["name"],
            checkpoint=spec["checkpoint"],
            sources=sources,
            gen_cfg=config["inference"],
            batch_size=batch_size,
        )

    final_predictions: List[str] = []
    for row_idx, source in enumerate(sources):
        per_model_top = []
        per_model_weights = []
        all_texts_for_rerank = []
        all_weights_for_rerank = []

        for spec in model_specs:
            row_candidates = all_candidates[spec["name"]][row_idx]
            top_text = row_candidates[0]["text"]
            per_model_top.append(top_text)
            per_model_weights.append(spec["norm_weight"])

            for cand in row_candidates:
                all_texts_for_rerank.append(cand["text"])
                all_weights_for_rerank.append(spec["norm_weight"])

        if len(model_specs) == 1:
            selected = per_model_top[0]
        elif n_best <= 1:
            selected = _weighted_vote(per_model_top, per_model_weights)
        else:
            selected = consensus_rerank(
                candidates=all_texts_for_rerank,
                model_weights=all_weights_for_rerank,
                bleu_weight=float(
                    config["inference"]["ensemble"]["rerank_weights"].get("bleu", 0.45)
                ),
                chrf_weight=float(
                    config["inference"]["ensemble"]["rerank_weights"].get("chrf", 0.45)
                ),
                length_weight=float(
                    config["inference"]["ensemble"]["rerank_weights"].get("length", 0.10)
                ),
            )

        if bool(config["inference"].get("enable_entity_repair", True)):
            selected = repair_named_entities(selected, source, lexicon=lexicon)
        final_predictions.append(canonicalize_text(selected, is_translation=True))

    if sample_submission is not None and not sample_submission.empty:
        submission_df = sample_submission.copy()
        submission_df[target_col] = final_predictions
    else:
        submission_df = pd.DataFrame({id_col: test_df[id_col], target_col: final_predictions})

    validate_submission_df(submission_df, test_df, id_col=id_col, target_col=target_col)

    output_root = config["global"]["output_dir"]
    os.makedirs(output_root, exist_ok=True)
    submission_path = os.path.join(output_root, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    metadata = {
        "submission_path": submission_path,
        "num_rows": len(submission_df),
        "models": model_specs,
        "n_best": n_best,
        "ensemble_method": config["inference"]["ensemble"].get("method", "weighted_vote"),
    }
    save_json(metadata, os.path.join(output_root, "inference_metadata.json"))
    return submission_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Past MT inference entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="",
        help="Path to training_summary.json",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.data_dir:
        config["global"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["global"]["output_dir"] = args.output_dir

    submission_path = run_inference(config, summary_path=args.summary)
    print(f"Submission written to {submission_path}")


if __name__ == "__main__":
    main()
