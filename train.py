import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from utils import (
    annotate_stress_slices,
    build_group_folds,
    build_named_entity_lexicon,
    build_sentence_level_train,
    canonicalize_text,
    competition_score,
    config_hash,
    filter_parallel_pairs,
    load_competition_data,
    load_yaml,
    mine_dictionary_pairs,
    mine_weak_parallel_data,
    save_json,
    set_seed,
)


def _prepare_master_training_frame(
    data_frames: Dict[str, pd.DataFrame], config: Dict[str, Any],
    ne_lexicon: Optional[set] = None,
) -> pd.DataFrame:
    sentence_df = build_sentence_level_train(
        train_df=data_frames["train"],
        sentence_meta_df=data_frames.get("sentence_meta"),
    )

    weak_df = pd.DataFrame(columns=["source", "target", "group_id", "origin"])
    if config["data"].get("use_weak_data", False):
        weak_df = mine_weak_parallel_data(
            published_df=data_frames.get("published_texts"),
            publications_df=data_frames.get("publications"),
            max_pairs=int(config["data"].get("weak_max_pairs", 2500)),
            low_ratio=float(config["data"].get("ratio_low", 0.12)),
            high_ratio=float(config["data"].get("ratio_high", 6.0)),
            ne_lexicon=ne_lexicon,
        )

    dict_df = pd.DataFrame(columns=["source", "target", "group_id", "origin"])
    if config["data"].get("use_dictionary_data", False):
        dict_df = mine_dictionary_pairs(
            ebl_dict_df=data_frames.get("ebl_dictionary"),
            max_pairs=int(config["data"].get("dict_max_pairs", 3000)),
        )

    merged = pd.concat([sentence_df, weak_df, dict_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)
    merged = filter_parallel_pairs(
        merged,
        min_src_tokens=int(config["data"].get("min_src_tokens", 2)),
        max_src_tokens=int(config["data"].get("max_src_tokens", 256)),
        min_tgt_tokens=int(config["data"].get("min_tgt_tokens", 2)),
        max_tgt_tokens=int(config["data"].get("max_tgt_tokens", 256)),
        ratio_low=float(config["data"].get("ratio_low", 0.12)),
        ratio_high=float(config["data"].get("ratio_high", 6.0)),
    )
    return merged


def _apply_stage_sampling(
    frame: pd.DataFrame,
    stage_cfg: Dict[str, Any],
    weak_origin: str = "weak_publications",
) -> pd.DataFrame:
    x = frame.copy()
    include_weak = stage_cfg.get("include_weak_data", False)
    weak_ratio = float(stage_cfg.get("weak_sample_ratio", 0.20))

    if not include_weak:
        x = x[x["origin"] != weak_origin]
    else:
        weak_part = x[x["origin"] == weak_origin]
        strong_part = x[x["origin"] != weak_origin]
        if not weak_part.empty and weak_ratio > 0.0:
            weak_n = min(len(weak_part), int(max(1, len(strong_part) * weak_ratio)))
            weak_part = weak_part.sample(n=weak_n, random_state=int(stage_cfg["seed"]))
            x = pd.concat([strong_part, weak_part], ignore_index=True)
        else:
            x = strong_part

    if stage_cfg.get("hard_replay", False):
        stress = annotate_stress_slices(x)
        hard = stress[stress["has_gap"] | stress["long_target"] | stress["named_entity_like"]]
        replay_factor = int(stage_cfg.get("hard_replay_factor", 2))
        if not hard.empty and replay_factor > 1:
            replay = pd.concat([hard] * (replay_factor - 1), ignore_index=True)
            x = pd.concat([x, replay[x.columns]], ignore_index=True)
    return x.sample(frac=1.0, random_state=int(stage_cfg["seed"])).reset_index(drop=True)


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_source_length: int,
    max_target_length: int,
    task_prefix: str = "",
) -> Dataset:
    def _map(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        sources = [task_prefix + s for s in batch["source"]]
        model_inputs = tokenizer(
            sources,
            truncation=True,
            max_length=max_source_length,
        )
        labels = tokenizer(
            batch["target"],
            truncation=True,
            max_length=max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(_map, batched=True, remove_columns=dataset.column_names)


def _build_compute_metrics(tokenizer: AutoTokenizer):
    def _compute(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [canonicalize_text(x, is_translation=True) for x in decoded_preds]
        decoded_labels = [canonicalize_text(x, is_translation=True) for x in decoded_labels]
        score = competition_score(decoded_preds, decoded_labels)
        return {"geo_mean": score}

    return _compute


def _train_single_stage(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    stage_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    task_prefix: str = "",
) -> Tuple[str, float]:
    train_ds = Dataset.from_pandas(train_df[["source", "target"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["source", "target"]], preserve_index=False)

    train_tok = _tokenize_dataset(
        train_ds,
        tokenizer,
        max_source_length=int(global_cfg["training"]["max_source_length"]),
        max_target_length=int(global_cfg["training"]["max_target_length"]),
        task_prefix=task_prefix,
    )
    val_tok = _tokenize_dataset(
        val_ds,
        tokenizer,
        max_source_length=int(global_cfg["training"]["max_source_length"]),
        max_target_length=int(global_cfg["training"]["max_target_length"]),
        task_prefix=task_prefix,
    )

    os.makedirs(output_dir, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy=global_cfg["training"].get("evaluation_strategy", "epoch"),
        save_strategy=global_cfg["training"].get("save_strategy", "epoch"),
        learning_rate=float(stage_cfg["learning_rate"]),
        per_device_train_batch_size=int(global_cfg["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(global_cfg["training"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(global_cfg["training"]["gradient_accumulation_steps"]),
        num_train_epochs=float(stage_cfg["epochs"]),
        warmup_ratio=float(stage_cfg["warmup_ratio"]),
        logging_steps=int(global_cfg["training"].get("logging_steps", 50)),
        save_total_limit=int(global_cfg["training"].get("save_total_limit", 3)),
        predict_with_generate=True,
        generation_max_length=int(global_cfg["training"]["generation_max_length"]),
        generation_num_beams=int(global_cfg["training"].get("eval_generation_num_beams",
                                    global_cfg["training"]["generation_num_beams"])),
        fp16=bool(global_cfg["training"].get("fp16", True)),
        load_best_model_at_end=True,
        metric_for_best_model="geo_mean",
        greater_is_better=True,
        label_smoothing_factor=float(stage_cfg.get("label_smoothing", 0.0)),
        optim=stage_cfg.get("optimizer", "adamw_torch"),
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=_build_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(
                    global_cfg["training"].get("early_stopping_patience", 2)
                )
            )
        ],
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    score = float(metrics.get("eval_geo_mean", -1.0))
    return output_dir, score


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(int(config["global"]["seed"]))
    data_dir = config["global"]["data_dir"]
    output_root = config["global"]["output_dir"]
    os.makedirs(output_root, exist_ok=True)

    frames = load_competition_data(data_dir)
    lexicon = build_named_entity_lexicon(frames.get("oa_lexicon"))

    master = _prepare_master_training_frame(frames, config, ne_lexicon=lexicon)
    master = build_group_folds(
        master,
        group_col="group_id",
        n_splits=int(config["cv"]["n_splits"]),
        seed=int(config["global"]["seed"]),
    )

    val_fold = int(config["cv"]["val_fold"])
    train_master = master[master["fold"] != val_fold].reset_index(drop=True)
    val_master = master[master["fold"] == val_fold].reset_index(drop=True)

    if config.get("debug", {}).get("smoke_max_rows"):
        smoke_n = int(config["debug"]["smoke_max_rows"])
        train_master = train_master.head(smoke_n).copy()
        val_master = val_master.head(max(20, smoke_n // 5)).copy()

    summary: Dict[str, Any] = {
        "config_hash": config_hash(config),
        "seed": int(config["global"]["seed"]),
        "val_fold": val_fold,
        "rows_train": len(train_master),
        "rows_val": len(val_master),
        "models": [],
    }

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        pretrained = model_cfg["pretrained_model_name"]
        task_prefix = model_cfg.get("task_prefix", "")
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)

        if model_cfg.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()

        model_entry = {"name": model_name, "pretrained": pretrained, "task_prefix": task_prefix, "stages": []}
        stage_train = train_master.copy()

        for stage in model_cfg["stages"]:
            stage_cfg = dict(stage)
            stage_cfg["seed"] = int(config["global"]["seed"])
            stage_cfg["optimizer"] = model_cfg.get("optimizer", "adamw_torch")
            stage_cfg["label_smoothing"] = stage.get(
                "label_smoothing", model_cfg.get("label_smoothing", 0.0)
            )

            stage_frame = _apply_stage_sampling(stage_train, stage_cfg)
            stage_out = os.path.join(output_root, model_name, stage["name"])
            checkpoint_dir, val_score = _train_single_stage(
                model=model,
                tokenizer=tokenizer,
                train_df=stage_frame,
                val_df=val_master,
                output_dir=stage_out,
                stage_cfg=stage_cfg,
                global_cfg=config,
                task_prefix=task_prefix,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
            if model_cfg.get("gradient_checkpointing", False):
                model.gradient_checkpointing_enable()

            model_entry["stages"].append(
                {
                    "stage": stage["name"],
                    "checkpoint_dir": checkpoint_dir,
                    "val_geo_mean": val_score,
                    "rows_train_stage": len(stage_frame),
                }
            )
        summary["models"].append(model_entry)

    summary_path = os.path.join(output_root, "training_summary.json")
    save_json(summary, summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Past MT training entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override config.global.data_dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override config.global.output_dir",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable 50-row smoke run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.data_dir:
        config["global"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["global"]["output_dir"] = args.output_dir
    if args.smoke:
        config.setdefault("debug", {})["smoke_max_rows"] = 50

    summary = run_training(config)
    print("Training complete.")
    print(pd.DataFrame(summary["models"]).to_string(index=False))
    print(f"Summary written to: {os.path.join(config['global']['output_dir'], 'training_summary.json')}")


if __name__ == "__main__":
    main()
