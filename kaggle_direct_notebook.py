"""
==========================================================================
Deep Past Challenge - Akkadian to English Translation
==========================================================================
TEK DOSYA Kaggle notebook scripti. Hicbir ek dosyaya ihtiyac duymaz.

ADIM 1 - Egitim (Internet ON, GPU T4 x2 veya P100):
    main(mode="full")   # ~6-8 saat, byt5-base + mt5-base ensemble

ADIM 2 - Weights'i Dataset olarak kaydet:
    Notebook'u Save Version yap -> Output'tan New Dataset olustur

ADIM 3 - Submission (Internet OFF):
    main_inference_only(weights_dir="/kaggle/input/DATASET_ADINIZ")

Hizli test (30 dk, dusuk skor):
    main(mode="quick")  # sadece byt5-small
==========================================================================
Iyilestirmeler (v2):
  - byt5-base + mt5-base  (small/flan-t5 yerine — cok daha buyuk modeller)
  - warmup + cosine LR + weight decay + label smoothing
  - gradient checkpointing (bellek tasarrufu)
  - as_target_tokenizer() deprecated fix
  - supplemental data (published_texts.csv) destegi
  - num_beams=10, no_repeat_ngram_size=4 (inference kalitesi)
  - early stopping patience=3
==========================================================================
"""

import os
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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

# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
OUTPUT_DIR = "/kaggle/working"
SEED = 42
MAX_SOURCE_LENGTH = 384   # Uzun Akkadca metinler icin arttirildi
MAX_TARGET_LENGTH = 200
TASK_PREFIX = "translate Akkadian to English: "

# Inference hyperparameters — kalite icin yuksek tutuldu
INFER_NUM_BEAMS = 10
INFER_LENGTH_PENALTY = 0.8
INFER_NO_REPEAT_NGRAM = 4
INFER_REPETITION_PENALTY = 1.3


# ============================================================================
# UTILITIES
# ============================================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_notebook_runtime() -> bool:
    if "ipykernel" in __import__("sys").modules:
        return True
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return False


def canonicalize_text(text: str, is_translation: bool = False) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    if is_translation:
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text
    text = text.lower()
    text = text.replace("s,", "\u1e63").replace("t,", "\u1e6d").replace("sz", "\u0161")
    text = re.sub(r"\[(x|\s*x\s*)+\]", " <big_gap> ", text)
    text = re.sub(r"\bx\b", " <gap> ", text)
    text = re.sub(r"(\s*<gap>\s*){2,}", " <big_gap> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def postprocess(text: str) -> str:
    cleaned = canonicalize_text(text, is_translation=True)
    # Model bazen prefix tekrarlayabilir
    cleaned = re.sub(r"^translate\s+akkadian\s+to\s+english:\s*", "", cleaned, flags=re.IGNORECASE)
    # Arka arkaya ayni kelime tekrari
    cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else "the text is fragmentary"


def competition_score(preds: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    if corpus_bleu is None or corpus_chrf is None:
        return {"bleu": 0.0, "chrf": 0.0, "geomean": 0.0}
    bleu = corpus_bleu(list(preds), [list(refs)]).score
    chrf = corpus_chrf(list(preds), [list(refs)], word_order=2).score
    geo = float(np.sqrt(max(bleu, 0.0) * max(chrf, 0.0)))
    return {"bleu": bleu, "chrf": chrf, "geomean": geo}


# ============================================================================
# DATA
# ============================================================================
def load_data(data_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    train.csv : oare_id, transliteration, translation  (dokuman duzeyinde)
    test.csv  : id, text_id, line_start, line_end, transliteration  (cumle duzeyinde)
    """
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    train_df["transliteration"] = train_df["transliteration"].fillna("")
    train_df["translation"] = train_df["translation"].fillna("")
    test_df["transliteration"] = test_df["transliteration"].fillna("")

    sample_sub_path = os.path.join(data_dir, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path) if os.path.exists(sample_sub_path) else None

    print(f"Train: {len(train_df)} rows, cols={list(train_df.columns)}")
    print(f"Test:  {len(test_df)} rows,  cols={list(test_df.columns)}")
    return train_df, test_df, sample_sub


def load_supplemental_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    published_texts.csv icinde ek paralel veri olabilir.
    transliteration + translation sutunlari varsa egitim setine ekle.
    """
    extra_rows = []
    for fname in ("published_texts.csv", "Sentences_Oare_FirstWord_LinNum.csv"):
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
            # Sutun adlarini normalize et
            cols = {c.lower().strip(): c for c in df.columns}
            src_col = next((cols[k] for k in ("transliteration", "translit", "source") if k in cols), None)
            tgt_col = next((cols[k] for k in ("translation", "translation_en", "target") if k in cols), None)
            if src_col and tgt_col:
                sub = df[[src_col, tgt_col]].rename(
                    columns={src_col: "transliteration", tgt_col: "translation"}
                ).dropna()
                sub = sub[sub["transliteration"].str.strip() != ""]
                sub = sub[sub["translation"].str.strip() != ""]
                extra_rows.append(sub)
                print(f"  Supplemental '{fname}': +{len(sub)} rows")
        except Exception as e:
            print(f"  Supplemental '{fname}' yuklenemedi: {e}")

    if extra_rows:
        return pd.concat(extra_rows, ignore_index=True)
    return pd.DataFrame(columns=["transliteration", "translation"])


def make_split(df: pd.DataFrame, seed: int = SEED, val_ratio: float = 0.15):
    """Group-based train/val split. train.csv'de oare_id kullanilir."""
    id_col = "oare_id" if "oare_id" in df.columns else "id"
    group = df[id_col].astype(str).str.extract(r"(^[^_]+)")[0].fillna(df[id_col].astype(str))
    uniq = group.unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_ratio))
    val_groups = set(uniq[:n_val])
    is_val = group.isin(val_groups)
    return df.loc[~is_val].reset_index(drop=True), df.loc[is_val].reset_index(drop=True)


# ============================================================================
# DATASET
# ============================================================================
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
        # as_target_tokenizer() deprecated — text_target= kullan
        enc = self.tokenizer(
            self.sources[idx],
            truncation=True,
            max_length=self.max_source_length,
        )
        labels = self.tokenizer(
            text_target=self.targets[idx],
            truncation=True,
            max_length=self.max_target_length,
        )
        enc["labels"] = labels["input_ids"]
        return enc


# ============================================================================
# TRAINING
# ============================================================================
def train_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int = SEED,
    use_gradient_checkpointing: bool = True,
) -> Tuple[str, float]:
    n_train = len(train_df)
    steps_per_epoch = max(1, n_train // (batch_size * grad_accum))
    warmup_steps = steps_per_epoch * 1  # ilk epoch warmup

    print(f"\n{'='*60}")
    print(f"Model : {model_name}")
    print(f"  epochs={epochs}, bs={batch_size}, ga={grad_accum}, eff_bs={batch_size*grad_accum}")
    print(f"  lr={lr}, warmup_steps={warmup_steps}")
    print(f"  train={n_train}, val={len(val_df)}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    def make_src(rows):
        return [TASK_PREFIX + canonicalize_text(x, False) for x in rows["transliteration"].tolist()]

    def make_tgt(rows):
        return [canonicalize_text(x, True) for x in rows["translation"].tolist()]

    train_ds = MTDataset(tokenizer, make_src(train_df), make_tgt(train_df),
                         MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)
    val_ds = MTDataset(tokenizer, make_src(val_df), make_tgt(val_df),
                       MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size * 2),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
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
        return competition_score(
            [canonicalize_text(x, True) for x in pred_txt],
            [canonicalize_text(x, True) for x in ref_txt],
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    metrics = trainer.evaluate(max_length=MAX_TARGET_LENGTH, num_beams=4)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    score = float(metrics.get("eval_geomean", 0.0))
    print(f"  -> val geomean = {score:.4f}  (bleu={metrics.get('eval_bleu',0):.2f}, chrf={metrics.get('eval_chrf',0):.2f})")
    return output_dir, score


# ============================================================================
# INFERENCE
# ============================================================================
def generate_predictions(
    checkpoint: str,
    sources: List[str],
    batch_size: int = 8,
) -> List[str]:
    """Bir checkpoint'ten yuksek kaliteli tahminler uret."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    all_preds: List[str] = []
    total_batches = (len(sources) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(sources), batch_size)):
        batch_sources = [TASK_PREFIX + s for s in sources[start: start + batch_size]]
        enc = tokenizer(
            batch_sources,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
            padding=True,
        )
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                num_beams=INFER_NUM_BEAMS,
                max_length=MAX_TARGET_LENGTH,
                length_penalty=INFER_LENGTH_PENALTY,
                no_repeat_ngram_size=INFER_NO_REPEAT_NGRAM,
                repetition_penalty=INFER_REPETITION_PENALTY,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        all_preds.extend([postprocess(x) for x in decoded])

        if (batch_idx + 1) % 20 == 0 or batch_idx == total_batches - 1:
            print(f"  inference batch {batch_idx+1}/{total_batches}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_preds


def predict_ensemble(
    model_dirs: List[str],
    weights: List[float],
    test_df: pd.DataFrame,
    out_csv: str,
) -> pd.DataFrame:
    """
    Coklu model + kaynak varyanti agirlikli ensemble.
    Kaynak varyantlari: canonical (agir), raw (hafif), no_determinatives (hafif)
    """
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    raw_sources = test_df["transliteration"].fillna("").astype(str).tolist()
    canonical = [canonicalize_text(x, False) for x in raw_sources]
    raw_lower = [str(x).strip().lower() for x in raw_sources]
    no_det = [s.replace("{", "").replace("}", "") for s in canonical]

    # (isim, agirlik, kaynak_listesi)
    variants: List[Tuple[str, float, List[str]]] = [
        ("canonical", 0.60, canonical),
        ("no_det",    0.25, no_det),
        ("raw",       0.15, raw_lower),
    ]
    vw = np.array([v[1] for v in variants], dtype=np.float64)
    vw = vw / vw.sum()

    # (model_idx, variant_idx, tahminler)
    all_preds: List[Tuple[int, int, List[str]]] = []

    for m_idx, mdir in enumerate(model_dirs):
        print(f"\n[Model {m_idx+1}/{len(model_dirs)}] {mdir}")
        for v_idx, (vname, _, vsources) in enumerate(variants):
            print(f"  variant={vname}")
            preds = generate_predictions(mdir, vsources)
            all_preds.append((m_idx, v_idx, preds))

    # Agirlikli oy birlestirme
    final_preds: List[str] = []
    for i in range(len(test_df)):
        pool: Dict[str, float] = defaultdict(float)
        for m_idx, v_idx, pv in all_preds:
            pool[pv[i]] += float(w[m_idx] * vw[v_idx])
        # En yuksek agirlikli, eger esit ise en kisa cumleyi sec
        best = sorted(pool.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
        final_preds.append(best)

    sub = pd.DataFrame({"id": test_df["id"], "translation": final_preds})
    sub.to_csv(out_csv, index=False)
    print(f"\nSubmission yazildi -> {out_csv} ({len(sub)} satir)")
    return sub


# ============================================================================
# CHECKPOINT UTILS
# ============================================================================
def is_valid_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_model = any(
        os.path.exists(os.path.join(path, f))
        for f in ("model.safetensors", "pytorch_model.bin")
    )
    has_tokenizer = any(
        os.path.exists(os.path.join(path, f))
        for f in ("tokenizer.json", "spiece.model", "sentencepiece.bpe.model", "tokenizer_config.json")
    )
    return has_config and has_model and has_tokenizer


def find_checkpoints(weights_dir: str) -> List[str]:
    """Gecerli HuggingFace checkpoint dizinlerini bul (2 seviye derin)."""
    found: List[str] = []
    if not os.path.isdir(weights_dir):
        return found
    for item in sorted(os.listdir(weights_dir)):
        p = os.path.join(weights_dir, item)
        if is_valid_checkpoint(p):
            found.append(p)
        elif os.path.isdir(p):
            for sub in sorted(os.listdir(p)):
                sp = os.path.join(p, sub)
                if is_valid_checkpoint(sp):
                    found.append(sp)
    return found


def copy_weights(ckpts: List[str], dest_root: str) -> None:
    os.makedirs(dest_root, exist_ok=True)
    for ckpt in ckpts:
        name = os.path.basename(ckpt)
        dest = os.path.join(dest_root, name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(ckpt, dest)
        print(f"  Kopyalandi: {name} -> {dest}")


# ============================================================================
# MAIN: FULL PIPELINE (Internet ON)
# ============================================================================
def main(
    mode: str = "full",
    data_dir: str = DATA_DIR,
    out_dir: str = os.path.join(OUTPUT_DIR, "akkadian_run"),
    seed: int = SEED,
) -> str:
    """
    Internet ON Kaggle notebook'ta calistir. Egitim + inference + weights kaydetme.

    mode="full"  : byt5-base (12 epoch) + mt5-base (10 epoch) — en yuksek skor
    mode="quick" : byt5-small (8 epoch)  — hizli test, ~1 saat
    """
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    train_df, test_df, _ = load_data(data_dir)

    # --- Supplemental veri yukle ---
    print("\nSupplemental veri aranıyor...")
    extra_df = load_supplemental_data(data_dir)

    # --- Train/val split (sadece ana train.csv'den) ---
    tr, va = make_split(train_df, seed=seed)
    print(f"Split: train={len(tr)}, val={len(va)}")

    # --- Supplemental veriyi train'e ekle (val'e degil) ---
    if len(extra_df) > 0:
        tr = pd.concat([tr, extra_df], ignore_index=True)
        print(f"  Supplemental eklendi -> train={len(tr)}")

    # --- Model tanimlari ---
    # (model_adi, epoch, batch_size, grad_accum, lr)
    if mode == "full":
        model_specs = [
            # byt5-base: karakter duzeyinde, Akkadca icin ideal
            ("google/byt5-base", 12, 2, 8, 8e-5),
            # mt5-base: cok dilli, cesitlilik saglar
            ("google/mt5-base", 10, 4, 4, 5e-5),
        ]
    else:  # quick
        model_specs = [
            ("google/byt5-small", 8, 8, 4, 2e-4),
        ]

    ckpts: List[str] = []
    scores: List[float] = []

    for model_name, epochs, bs, ga, lr in model_specs:
        safe_name = model_name.split("/")[-1].replace("-", "_")
        model_out = os.path.join(out_dir, safe_name)
        ckpt, score = train_one_model(
            model_name=model_name,
            train_df=tr,
            val_df=va,
            output_dir=model_out,
            epochs=epochs,
            batch_size=bs,
            grad_accum=ga,
            lr=lr,
            seed=seed,
        )
        ckpts.append(ckpt)
        scores.append(max(score, 1e-6))

    # --- Inference ---
    out_csv = os.path.join(OUTPUT_DIR, "submission.csv")
    predict_ensemble(ckpts, scores, test_df, out_csv)

    # --- Weights'i Dataset yukleme icin kopyala ---
    weights_upload = os.path.join(OUTPUT_DIR, "weights_upload")
    copy_weights(ckpts, weights_upload)
    print(f"\nWeights dizini: {weights_upload}")
    print("SONRAKI ADIM: Save Version -> Output -> New Dataset")

    return out_csv


# ============================================================================
# MAIN: INFERENCE ONLY (Internet OFF)
# ============================================================================
def main_inference_only(
    weights_dir: str = "/kaggle/input/akkadian-weights",
    data_dir: str = DATA_DIR,
) -> str:
    """
    Internet OFF submission notebook'ta calistir.
    weights_dir: Egitim notebook output'undan olusturduguno dataset path'i.
    """
    set_seed(SEED)

    # Checkpoint'leri bul
    candidates = [
        weights_dir,
        os.path.join(weights_dir, "weights_upload"),
        os.path.join(weights_dir, "akkadian_run"),
    ]
    resolved = None
    for c in candidates:
        if os.path.isdir(c) and find_checkpoints(c):
            resolved = c
            break

    if resolved is None and os.path.isdir("/kaggle/input"):
        print("Otomatik arama: /kaggle/input/")
        for d in sorted(os.listdir("/kaggle/input")):
            full = os.path.join("/kaggle/input", d)
            if not os.path.isdir(full):
                continue
            ckpts = find_checkpoints(full)
            if ckpts:
                resolved = full
                print(f"  Bulundu: {full} ({len(ckpts)} checkpoint)")
                break
            for sub in os.listdir(full)[:3]:
                sp = os.path.join(full, sub)
                if os.path.isdir(sp) and find_checkpoints(sp):
                    resolved = sp
                    break
            if resolved:
                break

    if resolved is None:
        raise FileNotFoundError(
            "Weights bulunamadi! Dataset'i notebook'a ekleyin.\n"
            f"Aranan: {candidates}"
        )

    ckpts = find_checkpoints(resolved)
    print(f"Weights: {resolved}")
    for c in ckpts:
        print(f"  {os.path.basename(c)}")

    _, test_df, _ = load_data(data_dir)

    out_csv = os.path.join(OUTPUT_DIR, "submission.csv")
    sub = predict_ensemble(ckpts, [1.0] * len(ckpts), test_df, out_csv)

    # Dogrulama
    assert len(sub) == len(test_df), f"Satir sayisi uyusmuyor: {len(sub)} != {len(test_df)}"
    assert sub["translation"].notna().all(), "Bos tahmin var!"
    assert (sub["translation"].str.strip() != "").all(), "Bos string var!"
    print("\nDogrulama BASARILI. Submit edebilirsiniz!")
    return out_csv


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    if is_notebook_runtime():
        print("Notebook modu.")
        print("  main(mode='full')  — egitim + inference")
        print("  main(mode='quick') — hizli test")
        print("  main_inference_only(weights_dir='...')  — sadece inference")
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Deep Past Akkadian-English MT")
        parser.add_argument("--mode", choices=["quick", "full", "inference"], default="full")
        parser.add_argument("--data-dir", default=DATA_DIR)
        parser.add_argument("--out-dir", default=os.path.join(OUTPUT_DIR, "akkadian_run"))
        parser.add_argument("--weights-dir", default="/kaggle/input/akkadian-weights")
        parser.add_argument("--seed", type=int, default=SEED)
        args, _ = parser.parse_known_args()

        if args.mode == "inference":
            main_inference_only(weights_dir=args.weights_dir, data_dir=args.data_dir)
        else:
            main(mode=args.mode, data_dir=args.data_dir, out_dir=args.out_dir, seed=args.seed)
