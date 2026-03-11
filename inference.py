import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

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


def _tokenizer_compat_kwargs(checkpoint: str) -> Dict[str, Any]:
    """
    Build compatibility kwargs for tokenizer loading.

    Some checkpoints contain `extra_special_tokens` as a list in
    tokenizer_config.json, while newer Transformers versions expect a dict.
    """
    cfg_path = os.path.join(checkpoint, "tokenizer_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}

    extra = cfg.get("extra_special_tokens")
    if isinstance(extra, list):
        return {"extra_special_tokens": {f"extra_id_{i}": tok for i, tok in enumerate(extra)}}
    return {}


def _normalize_weights(raw_weights: Sequence[float]) -> List[float]:
    if not raw_weights:
        return []
    total = sum(max(w, 1e-9) for w in raw_weights)
    return [max(w, 1e-9) / total for w in raw_weights]


def _postprocess_translation_candidate(text: str) -> str:
    """
    Light cleanup on generated translation text.

    This keeps canonicalization as the primary normalization while also
    removing common decoding artifacts (task prefix leakage, repeated tokens).
    """
    cleaned = canonicalize_text(text, is_translation=True)
    cleaned = re.sub(
        r"^translate\s+akkadian\s+to\s+english:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _closest_beam_group_count(num_beams: int, preferred_groups: int) -> int:
    groups = max(1, min(preferred_groups, num_beams))
    while groups > 1 and (num_beams % groups) != 0:
        groups -= 1
    return groups


def _build_generation_kwargs(gen_cfg: Dict[str, Any], n_best: int) -> Dict[str, Any]:
    """
    Build generation kwargs with optional diverse beam decoding.
    """
    num_beams = max(int(gen_cfg.get("num_beams", 4)), n_best)
    kwargs: Dict[str, Any] = {
        "num_beams": num_beams,
        "num_return_sequences": n_best,
        "max_length": int(gen_cfg.get("max_target_length", 160)),
        "length_penalty": float(gen_cfg.get("length_penalty", 1.0)),
        "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.0)),
        "no_repeat_ngram_size": int(gen_cfg.get("no_repeat_ngram_size", 0)),
        "early_stopping": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    diverse_cfg = gen_cfg.get("diverse_decoding", {}) or {}
    use_diverse = bool(diverse_cfg.get("enabled", False)) and n_best > 1 and num_beams > 1
    if use_diverse:
        preferred_groups = int(diverse_cfg.get("num_beam_groups", min(num_beams, n_best)))
        beam_groups = _closest_beam_group_count(num_beams, preferred_groups)
        if beam_groups > 1:
            kwargs["num_beam_groups"] = beam_groups
            kwargs["diversity_penalty"] = float(diverse_cfg.get("diversity_penalty", 0.3))
    return kwargs


def _build_source_variants(
    raw_sources: Sequence[str],
    inference_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    tta_cfg = inference_cfg.get("source_tta", {}) or {}
    enabled = bool(tta_cfg.get("enabled", False))
    if not enabled:
        return [{
            "name": "canonical",
            "weight": 1.0,
            "sources": [canonicalize_text(x, is_translation=False) for x in raw_sources],
        }]

    variants: List[Tuple[str, float, List[str]]] = []
    for idx, v in enumerate(tta_cfg.get("variants", [])):
        v_name = str(v.get("name", f"variant_{idx}"))
        v_type = str(v.get("type", "canonical"))
        v_weight = float(v.get("weight", 1.0))

        if v_type == "raw":
            srcs = [str(x).strip() for x in raw_sources]
        elif v_type == "no_determinatives":
            srcs = [
                canonicalize_text(x, is_translation=False).replace("{", "").replace("}", "")
                for x in raw_sources
            ]
        elif v_type == "biggap_to_gap":
            srcs = [
                canonicalize_text(x, is_translation=False).replace("<big_gap>", "<gap> <gap>")
                for x in raw_sources
            ]
        else:
            srcs = [canonicalize_text(x, is_translation=False) for x in raw_sources]
        variants.append((v_name, max(v_weight, 1e-9), srcs))

    if not variants:
        variants = [(
            "canonical",
            1.0,
            [canonicalize_text(x, is_translation=False) for x in raw_sources],
        )]

    norm = _normalize_weights([w for _, w, _ in variants])
    normalized_variants: List[Dict[str, Any]] = []
    for (name, _, srcs), w in zip(variants, norm):
        normalized_variants.append({"name": name, "weight": w, "sources": srcs})
    return normalized_variants


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
                    "task_prefix": m.get("task_prefix", ""),
                }
            )
        if specs:
            return specs

    for m in config["models"]:
        manual_ckpt = m.get("checkpoint")
        if manual_ckpt:
            specs.append({
                "name": m["name"],
                "checkpoint": manual_ckpt,
                "weight": 1.0,
                "task_prefix": m.get("task_prefix", ""),
            })
    return specs


def _generate_candidates_for_model(
    model_name: str,
    checkpoint: str,
    sources: Sequence[str],
    gen_cfg: Dict[str, Any],
    batch_size: int,
    task_prefix: str = "",
) -> List[List[Dict[str, float]]]:
    tok_kwargs = _tokenizer_compat_kwargs(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **tok_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    n_best = int(gen_cfg.get("n_best", 1))
    generation_kwargs = _build_generation_kwargs(gen_cfg, n_best=n_best)
    candidates: List[List[Dict[str, float]]] = []
    total_batches = (len(sources) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(sources), batch_size)):
        t0 = time.time()
        batch = [task_prefix + s for s in sources[start : start + batch_size]]
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
                **generation_kwargs,
            )

        decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        seq_scores = out.sequences_scores.detach().cpu().tolist()

        for i in range(len(batch)):
            row = []
            for j in range(n_best):
                idx = i * n_best + j
                row.append(
                    {
                        "text": _postprocess_translation_candidate(decoded[idx]),
                        "score": float(seq_scores[idx]),
                    }
                )
            candidates.append(row)

        elapsed = time.time() - t0
        done = min(start + batch_size, len(sources))
        remaining = total_batches - batch_idx - 1
        eta = elapsed * remaining
        print(f"  [{model_name}] batch {batch_idx + 1}/{total_batches} "
              f"({done}/{len(sources)} rows, {elapsed:.1f}s/batch, ETA {eta:.0f}s)")

    print(f"[inference] {model_name}: generated {len(candidates)} rows from {checkpoint}")
    return candidates


def _weighted_vote(row_candidates: List[str], row_weights: List[float]) -> str:
    score_by_text: Dict[str, float] = defaultdict(float)
    for text, weight in zip(row_candidates, row_weights):
        score_by_text[text] += weight
    ranked = sorted(score_by_text.items(), key=lambda kv: (-kv[1], len(kv[0])))
    return ranked[0][0]


def _majority_vote(row_candidates: List[str]) -> str:
    counts = defaultdict(int)
    for text in row_candidates:
        counts[text] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0])))
    return ranked[0][0]


def _length_vote(row_candidates: List[str], pick_longest: bool) -> str:
    non_empty = [c for c in row_candidates if c.strip()]
    pool = non_empty if non_empty else row_candidates
    if pick_longest:
        return max(pool, key=len)
    return min(pool, key=len)


def run_inference(config: Dict[str, Any], summary_path: str = "") -> str:
    set_seed(int(config["global"]["seed"]))
    frames = load_competition_data(config["global"]["data_dir"])
    test_df = frames["test"].copy()
    sample_submission = frames.get("sample_submission")
    lexicon = build_named_entity_lexicon(frames.get("oa_lexicon"))

    source_col = config["global"].get("source_column", "transliteration")
    target_col = config["global"].get("target_column", "translation")
    id_col = config["global"].get("id_column", "id")

    raw_sources = test_df[source_col].fillna("").astype(str).tolist()
    source_variants = _build_source_variants(raw_sources, config["inference"])
    default_sources = source_variants[0]["sources"]

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
    ensemble_cfg = config["inference"].get("ensemble", {})
    ensemble_method = str(ensemble_cfg.get("method", "weighted_vote")).lower()

    all_candidates: Dict[str, List[List[Dict[str, float]]]] = {}
    for spec in model_specs:
        for variant in source_variants:
            spec_key = f"{spec['name']}::{variant['name']}"
            all_candidates[spec_key] = _generate_candidates_for_model(
                model_name=spec_key,
                checkpoint=spec["checkpoint"],
                sources=variant["sources"],
                gen_cfg=config["inference"],
                batch_size=batch_size,
                task_prefix=spec.get("task_prefix", ""),
            )

    final_predictions: List[str] = []
    for row_idx, source in enumerate(default_sources):
        per_model_top = []
        per_model_weights = []
        all_texts_for_rerank = []
        all_weights_for_rerank = []
        all_beam_scores = []

        for spec in model_specs:
            best_variant_top_text = None
            best_variant_weight = -1.0
            for variant in source_variants:
                spec_key = f"{spec['name']}::{variant['name']}"
                row_candidates = all_candidates[spec_key][row_idx]
                combined_weight = spec["norm_weight"] * float(variant["weight"])
                if best_variant_top_text is None or combined_weight > best_variant_weight:
                    best_variant_top_text = row_candidates[0]["text"]
                    best_variant_weight = combined_weight

                for cand in row_candidates:
                    all_texts_for_rerank.append(cand["text"])
                    all_weights_for_rerank.append(combined_weight)
                    all_beam_scores.append(cand.get("score", 0.0))

            if best_variant_top_text is not None:
                per_model_top.append(best_variant_top_text)
                per_model_weights.append(max(best_variant_weight, 1e-9))

        if len(model_specs) == 1:
            selected = per_model_top[0]
        elif ensemble_method == "consensus_rerank":
            selected = consensus_rerank(
                candidates=all_texts_for_rerank,
                model_weights=all_weights_for_rerank,
                beam_scores=all_beam_scores,
                bleu_weight=float(
                    ensemble_cfg["rerank_weights"].get("bleu", 0.45)
                ),
                chrf_weight=float(
                    ensemble_cfg["rerank_weights"].get("chrf", 0.45)
                ),
                length_weight=float(
                    ensemble_cfg["rerank_weights"].get("length", 0.10)
                ),
                beam_score_weight=float(
                    ensemble_cfg["rerank_weights"].get("beam_score", 0.15)
                ),
            )
        elif ensemble_method == "majority_vote":
            selected = _majority_vote(per_model_top)
        elif ensemble_method == "longest":
            selected = _length_vote(per_model_top, pick_longest=True)
        elif ensemble_method == "shortest":
            selected = _length_vote(per_model_top, pick_longest=False)
        elif n_best <= 1:
            selected = _weighted_vote(per_model_top, per_model_weights)
        else:
            selected = consensus_rerank(
                candidates=all_texts_for_rerank,
                model_weights=all_weights_for_rerank,
                beam_scores=all_beam_scores,
                bleu_weight=float(
                    ensemble_cfg["rerank_weights"].get("bleu", 0.45)
                ),
                chrf_weight=float(
                    ensemble_cfg["rerank_weights"].get("chrf", 0.45)
                ),
                length_weight=float(
                    ensemble_cfg["rerank_weights"].get("length", 0.10)
                ),
                beam_score_weight=float(
                    ensemble_cfg["rerank_weights"].get("beam_score", 0.15)
                ),
            )

        if bool(config["inference"].get("enable_entity_repair", True)):
            selected = repair_named_entities(selected, source, lexicon=lexicon)
        final_predictions.append(_postprocess_translation_candidate(selected))

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
        "ensemble_method": ensemble_method,
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
