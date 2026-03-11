# Kaggle Tek Hücre (Copy/Paste) Çalıştırma

Aşağıdaki hücreyi **tek parça** halinde Kaggle notebook'una yapıştırıp çalıştır:

```python
%%bash
cat > /kaggle/working/kaggle_direct_notebook.py <<'PY'
import argparse
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
try:
    from sacrebleu import corpus_bleu, corpus_chrf
except Exception:
    corpus_bleu = None
    corpus_chrf = None
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize_text(text: str, is_translation: bool = False) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    if is_translation:
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text

    text = text.lower()
    text = text.replace("s,", "ṣ").replace("t,", "ṭ").replace("sz", "š")
    text = re.sub(r"\[(x|\s*x\s*)+\]", " <big_gap> ", text)
    text = re.sub(r"\bx\b", " <gap> ", text)
    text = re.sub(r"(\s*<gap>\s*){2,}", " <big_gap> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def competition_score(preds: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    if corpus_bleu is None or corpus_chrf is None:
        return {"bleu": 0.0, "chrf": 0.0, "geomean": 0.0}
    bleu = corpus_bleu(preds, [refs]).score
    chrf = corpus_chrf(preds, [refs], word_order=2).score
    geo = float(np.sqrt(max(bleu, 0.0) * max(chrf, 0.0)))
    return {"bleu": bleu, "chrf": chrf, "geomean": geo}


@dataclass
class MTDataset(torch.utils.data.Dataset):
    tokenizer: AutoTokenizer
    sources: List[str]
    targets: List[str]
    max_source_length: int
    max_target_length: int

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int):
        src = self.sources[idx]
        tgt = self.targets[idx]
        enc = self.tokenizer(src, truncation=True, max_length=self.max_source_length)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(tgt, truncation=True, max_length=self.max_target_length)
        enc["labels"] = labels["input_ids"]
        return enc


def make_split(df: pd.DataFrame, seed: int = 42, val_ratio: float = 0.15):
    id_col = "oare_id" if "oare_id" in df.columns else "id"
    group = df[id_col].astype(str).str.extract(r"(^[^_]+)")[0].fillna(df[id_col].astype(str))
    uniq = group.unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_ratio))
    val_groups = set(uniq[:n_val])
    is_val = group.isin(val_groups)
    return df.loc[~is_val].reset_index(drop=True), df.loc[is_val].reset_index(drop=True)


def train_one_model(model_name, train_df, val_df, output_dir, epochs, batch_size, lr, seed):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    prefix = "translate Akkadian to English: "
    train_src = [prefix + canonicalize_text(x, False) for x in train_df["transliteration"].tolist()]
    train_tgt = [canonicalize_text(x, True) for x in train_df["translation"].tolist()]
    val_src = [prefix + canonicalize_text(x, False) for x in val_df["transliteration"].tolist()]
    val_tgt = [canonicalize_text(x, True) for x in val_df["translation"].tolist()]

    train_ds = MTDataset(tokenizer, train_src, train_tgt, 320, 192)
    val_ds = MTDataset(tokenizer, val_src, val_tgt, 320, 192)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=192,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="geomean",
        greater_is_better=True,
        seed=seed,
    )

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred
        pred_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        ref_txt = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return competition_score([canonicalize_text(x, True) for x in pred_txt], [canonicalize_text(x, True) for x in ref_txt])

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    metrics = trainer.evaluate(max_length=192, num_beams=4)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir, metrics.get("eval_geomean", 0.0)


def predict_ensemble(model_dirs: List[str], weights: List[float], test_df: pd.DataFrame, out_csv: str):
    weights = np.array(weights, dtype=np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-9)

    base_sources = [canonicalize_text(x, False) for x in test_df["transliteration"].tolist()]
    variants = [
        ("canonical", 0.6, base_sources),
        ("raw", 0.2, [str(x).strip().lower() for x in test_df["transliteration"].tolist()]),
        ("no_det", 0.2, [s.replace("{", "").replace("}", "") for s in base_sources]),
    ]
    vweights = np.array([v[1] for v in variants], dtype=np.float64)
    vweights = vweights / vweights.sum()

    prefix = "translate Akkadian to English: "
    preds_by_model_variant = []
    for m_idx, mdir in enumerate(model_dirs):
        tok = AutoTokenizer.from_pretrained(mdir)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(mdir)
        mdl.eval()
        if torch.cuda.is_available():
            mdl = mdl.cuda()

        for v_idx, (_, _, var_sources) in enumerate(variants):
            all_preds = []
            batch_size = 16 if torch.cuda.is_available() else 4
            for st in range(0, len(var_sources), batch_size):
                batch = [prefix + x for x in var_sources[st: st + batch_size]]
                enc = tok(batch, return_tensors="pt", truncation=True, max_length=320, padding=True)
                if torch.cuda.is_available():
                    enc = {k: v.cuda() for k, v in enc.items()}
                with torch.no_grad():
                    out = mdl.generate(**enc, num_beams=5, max_length=192)
                all_preds.extend([canonicalize_text(x, True) for x in tok.batch_decode(out, skip_special_tokens=True)])
            preds_by_model_variant.append((m_idx, v_idx, all_preds))

    final_preds = []
    for i in range(len(test_df)):
        pool = {}
        for m_idx, v_idx, pv in preds_by_model_variant:
            txt = pv[i]
            pool[txt] = pool.get(txt, 0.0) + float(weights[m_idx] * vweights[v_idx])
        final_preds.append(sorted(pool.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0])

    pd.DataFrame({"id": test_df["id"], "translation": final_preds}).to_csv(out_csv, index=False)
    print(f"submission written -> {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--data-dir", default="/kaggle/input/deep-past-initiative-machine-translation")
    parser.add_argument("--out-dir", default="/kaggle/working/akkadian_run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    train_df["transliteration"] = train_df["transliteration"].fillna("")
    train_df["translation"] = train_df["translation"].fillna("")

    tr, va = make_split(train_df, seed=args.seed)
    model_specs = [("google/byt5-small", 5, 4, 2e-4), ("google/flan-t5-base", 3, 2, 1e-4)] if args.mode == "full" else [("google/byt5-small", 3, 8, 2e-4)]

    ckpts, scores = [], []
    for model_name, epochs, bs, lr in model_specs:
        out_dir = os.path.join(args.out_dir, model_name.split("/")[-1].replace("-", "_"))
        ckpt, score = train_one_model(model_name, tr, va, out_dir, epochs, bs, lr, args.seed)
        ckpts.append(ckpt)
        scores.append(max(float(score), 1e-6))

    predict_ensemble(ckpts, scores, test_df, "/kaggle/working/submission.csv")


if __name__ == "__main__":
    main()
PY
python /kaggle/working/kaggle_direct_notebook.py --mode full
```

Alternatif hızlı koşu:

```python
!python /kaggle/working/kaggle_direct_notebook.py --mode quick
```

Çıktı:

- `/kaggle/working/submission.csv`
