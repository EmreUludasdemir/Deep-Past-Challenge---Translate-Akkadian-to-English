"""
==========================================================================
NOTEBOOK 2: OFFLINE SUBMISSION (Internet OFF, GPU ON)
==========================================================================
Bu notebooku Kaggle'da Internet KAPALI olarak calistirin.
Onceden egitilmis model agirliklarini yukleyip sadece inference yapar.

Kaggle Notebook Ayarlari:
  - Accelerator: GPU T4 x2 (veya P100)
  - Internet: OFF  <-- ONEMLI!
  - Competition Data: deep-past-initiative-machine-translation (otomatik eklenir)
  - Ek Data: akkadian-weights (egitim notebook'undan olusturulan dataset)
  - Ek Data: deep-past-codev1 (kod dataset'i - opsiyonel, direkt paste de olur)

Kullanim:
  !python 2_offline_submission.py

veya notebook hucresine direkt yapistirin.
==========================================================================
"""

import json
import os
import re
import shutil
import sys
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =====================================================================
# YAPILANDIRMA - Ihtiyaca gore degistirin
# =====================================================================

# Yarismadan gelen veri
DATA_DIR = "/kaggle/input/deep-past-initiative-machine-translation"

# Egitim notebook'undan kaydedilen weights dataset
# Not: Dataset adi degistiyse buraya gercek path'i yazin
WEIGHTS_CANDIDATES = [
    "/kaggle/input/akkadian-weights",
    "/kaggle/input/akkadian-weights/weights_upload",
    "/kaggle/input/deep-past-weights",
]

# Kod dataset (opsiyonel - direkt paste kullaniyorsaniz gerek yok)
CODE_DATASET_CANDIDATES = [
    "/kaggle/input/deep-past-codev1",
    "/kaggle/input/deep-past-code-dataset",
]

# Inference ayarlari
NUM_BEAMS = 5
MAX_SOURCE_LENGTH = 320
MAX_TARGET_LENGTH = 192
BATCH_SIZE = 16  # GPU'ya gore ayarlayin

# =====================================================================
# YARDIMCI FONKSIYONLAR
# =====================================================================

def find_existing_path(candidates: list) -> str:
    for path in candidates:
        if os.path.isdir(path):
            return path
    return ""


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
    cleaned = re.sub(r"^translate\s+akkadian\s+to\s+english:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_valid_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_model = any(os.path.exists(os.path.join(path, f))
                    for f in ("model.safetensors", "pytorch_model.bin"))
    has_tokenizer = any(os.path.exists(os.path.join(path, f))
                        for f in ("tokenizer.json", "spiece.model",
                                  "sentencepiece.bpe.model", "tokenizer_config.json"))
    return has_config and has_model and has_tokenizer


def find_checkpoints(weights_dir: str) -> List[Tuple[str, str, float]]:
    """
    Weights dizinindeki tum gecerli checkpoint'lari bul.
    Returns: [(model_name, checkpoint_path, weight), ...]
    """
    found = []

    # Once training_summary.json varsa onu kullan
    summary_path = os.path.join(weights_dir, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        for m in summary.get("models", []):
            name = m["name"]
            stages = m.get("stages", [])
            if not stages:
                continue
            best = max(stages, key=lambda s: s.get("val_geo_mean", -1.0))
            # Checkpoint path'i weights_dir icindeki model klasorune isaret etmeli
            ckpt = os.path.join(weights_dir, name)
            if is_valid_checkpoint(ckpt):
                found.append((name, ckpt, max(best.get("val_geo_mean", 1.0), 1e-6)))
                continue
            # Eger orijinal path hala mevcutsa (ayni runtime ise)
            orig = best.get("checkpoint_dir", "")
            if orig and is_valid_checkpoint(orig):
                found.append((name, orig, max(best.get("val_geo_mean", 1.0), 1e-6)))
        if found:
            return found

    # Summary yoksa dizini tara
    for item in sorted(os.listdir(weights_dir)):
        item_path = os.path.join(weights_dir, item)
        if is_valid_checkpoint(item_path):
            found.append((item, item_path, 1.0))

    if not found:
        # Bir seviye daha derine bak
        for sub in sorted(os.listdir(weights_dir)):
            sub_path = os.path.join(weights_dir, sub)
            if os.path.isdir(sub_path):
                for item in sorted(os.listdir(sub_path)):
                    item_path = os.path.join(sub_path, item)
                    if is_valid_checkpoint(item_path):
                        found.append((item, item_path, 1.0))

    return found


# =====================================================================
# INFERENCE
# =====================================================================

def generate_predictions(
    checkpoint: str,
    sources: List[str],
    task_prefix: str = "translate Akkadian to English: ",
    num_beams: int = NUM_BEAMS,
    batch_size: int = BATCH_SIZE,
) -> List[str]:
    """Tek bir modelden tum test verisi icin tahmin uret."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    all_preds = []
    total = (len(sources) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(sources), batch_size)):
        batch = [task_prefix + s for s in sources[start:start + batch_size]]
        enc = tokenizer(
            batch,
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
                num_beams=num_beams,
                max_length=MAX_TARGET_LENGTH,
                length_penalty=1.2,
                repetition_penalty=1.2,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        all_preds.extend([postprocess(x) for x in decoded])

        if (batch_idx + 1) % 10 == 0 or batch_idx == total - 1:
            print(f"  Batch {batch_idx + 1}/{total} tamamlandi")

    # GPU bellegini temizle
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_preds


def weighted_ensemble(
    all_model_preds: List[List[str]],
    weights: List[float],
    n_rows: int,
) -> List[str]:
    """Birden fazla modelin tahminlerini agirlikli oylama ile birlestir."""
    norm_w = np.array(weights, dtype=np.float64)
    norm_w = norm_w / norm_w.sum()

    final = []
    for i in range(n_rows):
        pool: Dict[str, float] = defaultdict(float)
        for m_idx, preds in enumerate(all_model_preds):
            pool[preds[i]] += norm_w[m_idx]
        # En yuksek skorlu, esitlikte en kisa
        best = sorted(pool.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
        final.append(best)
    return final


def source_variant_ensemble(
    checkpoint: str,
    raw_sources: List[str],
    task_prefix: str = "translate Akkadian to English: ",
) -> List[str]:
    """Tek model icin 3 kaynak varyanti ile TTA benzeri ensemble."""
    canonical = [canonicalize_text(x, False) for x in raw_sources]
    raw = [str(x).strip().lower() for x in raw_sources]
    no_det = [s.replace("{", "").replace("}", "") for s in canonical]

    variants = [
        (canonical, 0.6),
        (raw, 0.2),
        (no_det, 0.2),
    ]

    all_variant_preds = []
    variant_weights = []
    for srcs, w in variants:
        preds = generate_predictions(checkpoint, srcs, task_prefix)
        all_variant_preds.append(preds)
        variant_weights.append(w)

    return weighted_ensemble(all_variant_preds, variant_weights, len(raw_sources))


# =====================================================================
# ANA AKIS
# =====================================================================

def main():
    print("=" * 60)
    print("OFFLINE SUBMISSION - Internet OFF")
    print("=" * 60)

    # 1. Veri yukle
    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test verisi bulunamadi: {test_path}\n"
            "Competition data eklenmemis. Notebook ayarlarindan ekleyin."
        )

    test_df = pd.read_csv(test_path)
    print(f"Test verisi: {len(test_df)} satir")

    sample_sub = None
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)

    # 2. Weights dizinini bul
    weights_dir = find_existing_path(WEIGHTS_CANDIDATES)
    if not weights_dir:
        # Tum /kaggle/input/ icinde ara
        input_dir = "/kaggle/input"
        if os.path.isdir(input_dir):
            print(f"\nMevcut input dizinleri:")
            for item in sorted(os.listdir(input_dir)):
                print(f"  /kaggle/input/{item}/")
                sub_path = os.path.join(input_dir, item)
                if os.path.isdir(sub_path):
                    for sub in sorted(os.listdir(sub_path))[:5]:
                        print(f"    {sub}")
        raise FileNotFoundError(
            "Weights dataset bulunamadi!\n"
            "Notebook ayarlarindan weights dataset'ini ekleyin.\n"
            f"Aranan dizinler: {WEIGHTS_CANDIDATES}"
        )

    print(f"Weights dizini: {weights_dir}")

    # 3. Checkpoint'lari bul
    checkpoints = find_checkpoints(weights_dir)
    if not checkpoints:
        print(f"\nDizin icerigi ({weights_dir}):")
        for item in sorted(os.listdir(weights_dir)):
            item_path = os.path.join(weights_dir, item)
            if os.path.isdir(item_path):
                files = os.listdir(item_path)[:5]
                print(f"  {item}/ -> {files}")
            else:
                print(f"  {item}")
        raise FileNotFoundError(
            f"Gecerli checkpoint bulunamadi: {weights_dir}\n"
            "Her model klasorunde config.json, model.safetensors ve tokenizer dosyalari olmali."
        )

    print(f"\nBulunan modeller ({len(checkpoints)}):")
    for name, path, weight in checkpoints:
        print(f"  {name}: {path} (weight={weight:.4f})")

    # 4. Kaynaklari hazirla
    raw_sources = test_df["transliteration"].fillna("").astype(str).tolist()

    # 5. Inference
    if len(checkpoints) == 1:
        # Tek model - source variant ensemble ile
        name, ckpt, _ = checkpoints[0]
        print(f"\nTek model ile source-variant ensemble: {name}")
        final_preds = source_variant_ensemble(ckpt, raw_sources)
    else:
        # Coklu model - her model icin canonical + weighted ensemble
        print(f"\n{len(checkpoints)} model ile ensemble")
        canonical_sources = [canonicalize_text(x, False) for x in raw_sources]

        all_model_preds = []
        model_weights = []
        for name, ckpt, weight in checkpoints:
            print(f"\nModel: {name}")
            preds = generate_predictions(ckpt, canonical_sources)
            all_model_preds.append(preds)
            model_weights.append(weight)

        final_preds = weighted_ensemble(all_model_preds, model_weights, len(test_df))

    # 6. Submission olustur
    if sample_sub is not None and not sample_sub.empty:
        submission = sample_sub.copy()
        submission["translation"] = final_preds
    else:
        submission = pd.DataFrame({
            "id": test_df["id"],
            "translation": final_preds,
        })

    # Bos satirlari kontrol et
    empty_mask = submission["translation"].isna() | (submission["translation"].str.strip() == "")
    if empty_mask.any():
        print(f"UYARI: {empty_mask.sum()} bos tahmin var, fallback uygulaniyor")
        submission.loc[empty_mask, "translation"] = "the text is damaged"

    out_path = "/kaggle/working/submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"SUBMISSION YAZILDI: {out_path}")
    print(f"Satir sayisi: {len(submission)}")
    print(f"{'=' * 60}")
    print(submission.head(10).to_string())

    # Dogrulama
    assert len(submission) == len(test_df), \
        f"Satir sayisi uyusmuyor: {len(submission)} != {len(test_df)}"
    assert submission["id"].nunique() == len(test_df), "Tekrar eden id var!"
    assert not submission["translation"].isna().any(), "NaN tahmin var!"
    print("\nDogrulama BASARILI - submit edebilirsiniz!")


if __name__ == "__main__":
    main()
