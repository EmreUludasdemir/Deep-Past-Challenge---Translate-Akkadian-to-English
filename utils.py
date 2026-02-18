import argparse
import hashlib
import json
import math
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import sacrebleu
except ImportError:  # pragma: no cover
    sacrebleu = None


ASCII_TO_DIACRITIC = {
    "sz": "š",
    "SZ": "Š",
    "s,": "ṣ",
    "S,": "Ṣ",
    "t,": "ṭ",
    "T,": "Ṭ",
    "h,": "ḫ",
    "H,": "Ḫ",
    "a2": "á",
    "A2": "Á",
    "a3": "à",
    "A3": "À",
    "e2": "é",
    "E2": "É",
    "e3": "è",
    "E3": "È",
    "i2": "í",
    "I2": "Í",
    "i3": "ì",
    "I3": "Ì",
    "u2": "ú",
    "U2": "Ú",
    "u3": "ù",
    "U3": "Ù",
}

DETERMINATIVE_REPLACEMENTS = {
    "{d}": "{d}",
    "{ki}": "{ki}",
    "{lu₂}": "{lu2}",
    "{e₂}": "{e2}",
    "{uru}": "{uru}",
    "{kur}": "{kur}",
    "{mi}": "{mi}",
    "{m}": "{m}",
    "{geš}": "{gesh}",
    "{ĝeš}": "{gesh}",
    "{tug₂}": "{tug2}",
    "{dub}": "{dub}",
    "{id₂}": "{id2}",
    "{mušen}": "{mushen}",
    "{na₄}": "{na4}",
    "{kuš}": "{kush}",
    "{u₂}": "{u2}",
}

TEXT_FILE_MAP = {
    "train": "train.csv",
    "test": "test.csv",
    "sample_submission": "sample_submission.csv",
    "published_texts": "published_texts.csv",
    "publications": "publications.csv",
    "bibliography": "bibliography.csv",
    "oa_lexicon": "OA_Lexicon_eBL.csv",
    "ebl_dictionary": "eBL_Dictionary.csv",
    "resources": "resources.csv",
    "sentence_meta": "Sentences_Oare_FirstWord_LinNum.csv",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def compute_bleu_chrf(
    predictions: Sequence[str], references: Sequence[str]
) -> Tuple[float, float]:
    if sacrebleu is None:
        raise ImportError(
            "sacrebleu is required for BLEU/chrF metric computation. "
            "Install with `pip install sacrebleu`."
        )
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length.")
    bleu = sacrebleu.corpus_bleu(list(predictions), [list(references)]).score
    chrfpp = sacrebleu.corpus_chrf(
        list(predictions), [list(references)], word_order=2
    ).score
    return bleu, chrfpp


def competition_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    bleu, chrfpp = compute_bleu_chrf(predictions, references)
    return float(math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0)))


def run_metric_fixture() -> Dict[str, float]:
    refs = [
        "Send a copy of this letter to every colony.",
        "From this day on, whoever buys tin must pay in silver.",
    ]
    preds = [
        "Send a copy of this letter to every colony.",
        "From this day on, whoever buys tin must pay in silver.",
    ]
    bleu, chrfpp = compute_bleu_chrf(preds, refs)
    score = competition_score(preds, refs)
    expected = math.sqrt(bleu * chrfpp)
    if abs(score - expected) > 1e-12:
        raise AssertionError("Metric wrapper mismatch.")
    return {"bleu": bleu, "chrfpp": chrfpp, "geometric_mean": score}


def normalize_unicode(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u2019", "'").replace("\u02be", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_diacritics(text: str) -> str:
    text = normalize_unicode(text)
    for src, tgt in ASCII_TO_DIACRITIC.items():
        text = re.sub(rf"(?<!\w){re.escape(src)}(?!\w)", tgt, text)
    text = text.replace("ḫ", "h").replace("Ḫ", "H")
    return text


def normalize_determinatives(text: str) -> str:
    for src, tgt in DETERMINATIVE_REPLACEMENTS.items():
        text = text.replace(src, tgt)
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)
    return text


def _gap_token_from_match(fragment: str) -> str:
    compact = re.sub(r"\s+", "", fragment)
    x_count = len(re.findall(r"x", compact, flags=re.IGNORECASE))
    if x_count > 1 or "..." in compact or "…" in compact:
        return "<big_gap>"
    return "<gap>"


def normalize_gap_markers(text: str) -> str:
    text = text.replace("…", " ... ")
    text = re.sub(r"<\s*gap\s*>", "<gap>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*big_gap\s*>", "<big_gap>", text, flags=re.IGNORECASE)

    def _replace_bracketed(match: re.Match) -> str:
        return _gap_token_from_match(match.group(0))

    text = re.sub(r"\[(?:\s*x\s*)+\]", _replace_bracketed, text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\.{3,}\s*\]", "<big_gap>", text)
    text = re.sub(r"\[\s*…+\s*\]", "<big_gap>", text)
    text = re.sub(r"(?<!\w)(x(?:\s+x)+)(?!\w)", "<big_gap>", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)x(?!\w)", "<gap>", text, flags=re.IGNORECASE)
    text = re.sub(r"\.{3,}", "<big_gap>", text)

    for _ in range(4):
        text = re.sub(r"<gap>\s+<big_gap>", "<big_gap>", text)
        text = re.sub(r"<big_gap>\s+<gap>", "<big_gap>", text)
        text = re.sub(r"<big_gap>\s+<big_gap>", "<big_gap>", text)
        text = re.sub(r"<gap>\s+<gap>", "<big_gap>", text)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_text(
    text: str,
    is_translation: bool = False,
    normalize_gaps: bool = True,
    normalize_det: bool = True,
) -> str:
    text = normalize_unicode(text)
    text = normalize_diacritics(text)
    if normalize_det:
        text = normalize_determinatives(text)
    if normalize_gaps:
        text = normalize_gap_markers(text)
    if is_translation:
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def split_english_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;:])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return [text.strip()]
    return parts


def split_transliteration_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\n", " ")
    parts = re.split(r"\s+(?=\d+''?|\d+'\b)|(?<=[.;:])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return [text.strip()]
    return parts


def align_document_pair(transliteration: str, translation: str) -> List[Tuple[str, str]]:
    src_sents = split_transliteration_sentences(transliteration)
    tgt_sents = split_english_sentences(translation)
    if not src_sents or not tgt_sents:
        return []

    if len(src_sents) == len(tgt_sents):
        return list(zip(src_sents, tgt_sents))

    if len(src_sents) > len(tgt_sents):
        ratio = len(src_sents) / len(tgt_sents)
        pairs = []
        for i, src in enumerate(src_sents):
            tgt_idx = min(int(i / ratio), len(tgt_sents) - 1)
            pairs.append((src, tgt_sents[tgt_idx]))
        return pairs

    ratio = len(tgt_sents) / len(src_sents)
    pairs = []
    for i, tgt in enumerate(tgt_sents):
        src_idx = min(int(i / ratio), len(src_sents) - 1)
        pairs.append((src_sents[src_idx], tgt))
    return pairs


def is_likely_english(text: str) -> bool:
    if not text:
        return False
    sample = text.lower()
    alpha = sum(ch.isalpha() for ch in sample)
    if alpha == 0:
        return False
    ascii_alpha = sum(ch.isalpha() and ord(ch) < 128 for ch in sample)
    if ascii_alpha / alpha < 0.8:
        return False
    stop_hits = sum(
        token in sample
        for token in [" the ", " and ", " of ", " to ", " in ", " from ", " with "]
    )
    return stop_hits >= 2


def length_ratio_ok(src: str, tgt: str, low: float = 0.12, high: float = 6.0) -> bool:
    s = max(len(src.split()), 1)
    t = max(len(tgt.split()), 1)
    ratio = t / s
    return low <= ratio <= high


def load_competition_data(data_dir: str) -> Dict[str, Optional[pd.DataFrame]]:
    out: Dict[str, Optional[pd.DataFrame]] = {}
    for key, file_name in TEXT_FILE_MAP.items():
        path = os.path.join(data_dir, file_name)
        if os.path.exists(path):
            out[key] = pd.read_csv(path)
        else:
            out[key] = None
    return out


def _best_group_column(df: pd.DataFrame) -> str:
    for col in ["oare_id", "text_id", "id"]:
        if col in df.columns:
            return col
    return df.columns[0]


def build_sentence_level_train(
    train_df: pd.DataFrame,
    sentence_meta_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if train_df is None or train_df.empty:
        return pd.DataFrame(columns=["source", "target", "group_id", "origin"])

    group_col = _best_group_column(train_df)
    for _, row in train_df.iterrows():
        src = str(row.get("transliteration", "") or "")
        tgt = str(row.get("translation", "") or "")
        group_id = str(row.get(group_col, ""))
        aligned = align_document_pair(src, tgt)
        if not aligned:
            continue
        for src_sent, tgt_sent in aligned:
            rows.append(
                {
                    "source": canonicalize_text(src_sent, is_translation=False),
                    "target": canonicalize_text(tgt_sent, is_translation=True),
                    "group_id": group_id,
                    "origin": "train_sentence",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["source", "target"])
    out = out[out["source"].str.len() > 0]
    out = out[out["target"].str.len() > 0]
    return out.reset_index(drop=True)


def mine_weak_parallel_data(
    published_df: Optional[pd.DataFrame],
    publications_df: Optional[pd.DataFrame],
    max_pairs: int = 2500,
    low_ratio: float = 0.12,
    high_ratio: float = 6.0,
    ne_lexicon: Optional[set] = None,
) -> pd.DataFrame:
    if published_df is None or publications_df is None:
        return pd.DataFrame(columns=["source", "target", "group_id", "origin"])
    if published_df.empty or publications_df.empty:
        return pd.DataFrame(columns=["source", "target", "group_id", "origin"])

    rows: List[Dict[str, Any]] = []
    published = published_df.copy()
    publications = publications_df.copy()

    pub_key_col = "oare_id" if "oare_id" in published.columns else published.columns[0]
    src_col = "transliteration" if "transliteration" in published.columns else None
    if src_col is None:
        return pd.DataFrame(columns=["source", "target", "group_id", "origin"])

    text_col = "page_text" if "page_text" in publications.columns else None
    if text_col is None:
        return pd.DataFrame(columns=["source", "target", "group_id", "origin"])

    doc_to_src = {}
    for _, row in published.iterrows():
        group_id = str(row.get(pub_key_col, ""))
        src = canonicalize_text(str(row.get(src_col, "") or ""), is_translation=False)
        if group_id and src:
            doc_to_src[group_id] = src

    for _, row in publications.iterrows():
        blob = str(row.get(text_col, "") or "")
        if not blob:
            continue
        candidates = split_english_sentences(blob)
        candidates = [c for c in candidates if is_likely_english(c)]
        if not candidates:
            continue
        blob_key_hits = [k for k in doc_to_src.keys() if k and k in blob]
        if not blob_key_hits:
            continue
        for group_id in blob_key_hits[:3]:
            src = doc_to_src[group_id]
            src_entities = extract_source_named_entities(src, lexicon=ne_lexicon) if ne_lexicon else []
            for sent in candidates[:4]:
                tgt = canonicalize_text(sent, is_translation=True)
                if not length_ratio_ok(src, tgt, low=low_ratio, high=high_ratio):
                    continue
                if src_entities:
                    tgt_lower = tgt.lower()
                    if not any(e.lower() in tgt_lower for e in src_entities):
                        continue
                rows.append(
                    {
                        "source": src,
                        "target": tgt,
                        "group_id": group_id,
                        "origin": "weak_publications",
                    }
                )
                if len(rows) >= max_pairs:
                    break
            if len(rows) >= max_pairs:
                break
        if len(rows) >= max_pairs:
            break

    if not rows:
        return pd.DataFrame(columns=["source", "target", "group_id", "origin"])

    out = pd.DataFrame(rows).drop_duplicates(subset=["source", "target"]).reset_index(
        drop=True
    )
    return out


def mine_dictionary_pairs(
    ebl_dict_df: Optional[pd.DataFrame],
    max_pairs: int = 3000,
) -> pd.DataFrame:
    """Extract word/phrase-level parallel pairs from eBL Dictionary."""
    empty = pd.DataFrame(columns=["source", "target", "group_id", "origin"])
    if ebl_dict_df is None or ebl_dict_df.empty:
        return empty

    src_col = next(
        (c for c in ebl_dict_df.columns if c.lower() in
         ("form", "word", "transliteration", "lemma", "cf")),
        None,
    )
    tgt_col = next(
        (c for c in ebl_dict_df.columns if c.lower() in
         ("meaning", "translation", "definition", "english", "gw", "guide_word")),
        None,
    )
    if not src_col or not tgt_col:
        return empty

    rows: List[Dict[str, Any]] = []
    for _, row in ebl_dict_df.iterrows():
        src = canonicalize_text(str(row.get(src_col, "") or ""), is_translation=False)
        tgt = canonicalize_text(str(row.get(tgt_col, "") or ""), is_translation=True)
        if not src or not tgt or len(src) < 2 or len(tgt) < 2:
            continue
        rows.append({
            "source": src,
            "target": tgt,
            "group_id": f"dict_{len(rows)}",
            "origin": "dictionary",
        })
        if len(rows) >= max_pairs:
            break

    if not rows:
        return empty
    return pd.DataFrame(rows).drop_duplicates(subset=["source", "target"]).reset_index(drop=True)


def build_named_entity_lexicon(oa_lexicon_df: Optional[pd.DataFrame]) -> set:
    if oa_lexicon_df is None or oa_lexicon_df.empty:
        return set()
    candidate_cols = [c for c in ["form", "norm", "lexeme"] if c in oa_lexicon_df.columns]
    if not candidate_cols:
        return set()

    ne_mask = None
    if "type" in oa_lexicon_df.columns:
        ne_mask = oa_lexicon_df["type"].astype(str).str.upper().isin({"PN", "GN"})
    if ne_mask is None:
        ne_mask = pd.Series([True] * len(oa_lexicon_df))
    subset = oa_lexicon_df.loc[ne_mask]

    lex = set()
    for col in candidate_cols:
        values = subset[col].dropna().astype(str).tolist()
        for v in values:
            tok = canonicalize_text(v, is_translation=False)
            if tok:
                lex.add(tok.lower())
    return lex


def extract_source_named_entities(source: str, lexicon: Optional[set] = None) -> List[str]:
    tokens = re.findall(r"[A-Za-zŠšṢṣṬṭÁÀÉÈÍÌÚÙ0-9'-]+", source)
    entities = []
    for tok in tokens:
        cleaned = tok.strip("-").lower()
        if not cleaned:
            continue
        if lexicon and cleaned in lexicon:
            entities.append(tok.strip("-"))
            continue
        if any(ch.isupper() for ch in tok) and len(tok) > 2:
            entities.append(tok.strip("-"))
    return list(dict.fromkeys(entities))


def repair_named_entities(
    prediction: str,
    source: str,
    lexicon: Optional[set] = None,
    max_append: int = 2,
) -> str:
    prediction = normalize_unicode(prediction)
    entities = extract_source_named_entities(source, lexicon=lexicon)
    if not entities:
        return prediction
    repaired = prediction
    lower_pred = f" {repaired.lower()} "

    missing = []
    for ent in entities:
        ent_lower = ent.lower()
        if f" {ent_lower} " in lower_pred:
            continue
        if ent_lower in repaired.lower():
            continue
        missing.append(ent)

    if not missing or len(repaired.split()) <= 2:
        return repaired

    to_add = missing[:max_append]
    stripped = repaired.rstrip(".,:;!? ")
    repaired = stripped + ", " + ", ".join(to_add) + "."
    return re.sub(r"\s+", " ", repaired).strip()


def filter_parallel_pairs(
    df: pd.DataFrame,
    min_src_tokens: int = 2,
    max_src_tokens: int = 256,
    min_tgt_tokens: int = 2,
    max_tgt_tokens: int = 256,
    ratio_low: float = 0.12,
    ratio_high: float = 6.0,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["source", "target", "group_id", "origin"])

    x = df.copy()
    x["src_len"] = x["source"].astype(str).str.split().str.len()
    x["tgt_len"] = x["target"].astype(str).str.split().str.len()
    x = x[(x["src_len"] >= min_src_tokens) & (x["src_len"] <= max_src_tokens)]
    x = x[(x["tgt_len"] >= min_tgt_tokens) & (x["tgt_len"] <= max_tgt_tokens)]
    x["len_ratio"] = x["tgt_len"] / x["src_len"].clip(lower=1)
    x = x[(x["len_ratio"] >= ratio_low) & (x["len_ratio"] <= ratio_high)]
    x = x.drop(columns=["src_len", "tgt_len", "len_ratio"]).reset_index(drop=True)
    return x


def build_group_folds(
    df: pd.DataFrame,
    group_col: str = "group_id",
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    x = df.copy().reset_index(drop=True)
    groups = x[group_col].astype(str).values

    try:
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=n_splits)
        fold = np.zeros(len(x), dtype=int)
        for fold_id, (_, val_idx) in enumerate(splitter.split(x, groups=groups)):
            fold[val_idx] = fold_id
    except Exception:
        rng = np.random.default_rng(seed)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        fold_map = {g: i % n_splits for i, g in enumerate(unique_groups)}
        fold = np.array([fold_map[g] for g in groups], dtype=int)

    x["fold"] = fold
    return x


def annotate_stress_slices(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["has_gap"] = x["source"].str.contains("<gap>|<big_gap>", regex=True)
    x["long_target"] = x["target"].str.split().str.len() >= 40
    x["named_entity_like"] = x["source"].str.contains(
        r"[A-ZŠṢṬÁÉÍÚ].*[a-zšṣṭáéíú]|{[a-z0-9]+}", regex=True
    )
    return x


def sentence_bleu_proxy(candidate: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    if sacrebleu is None:
        return 0.0
    return float(sacrebleu.sentence_bleu(candidate, list(references)).score)


def sentence_chrf_proxy(candidate: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    if sacrebleu is None:
        return 0.0
    scores = []
    for ref in references:
        score = sacrebleu.sentence_chrf(candidate, [ref], word_order=2).score
        scores.append(score)
    return float(np.mean(scores))


def consensus_rerank(
    candidates: Sequence[str],
    model_weights: Optional[Sequence[float]] = None,
    beam_scores: Optional[Sequence[float]] = None,
    bleu_weight: float = 0.45,
    chrf_weight: float = 0.45,
    length_weight: float = 0.10,
    beam_score_weight: float = 0.15,
) -> str:
    if not candidates:
        return ""
    unique = list(dict.fromkeys([c.strip() for c in candidates if c.strip()]))
    if len(unique) == 1:
        return unique[0]

    if model_weights is None:
        model_weights = [1.0] * len(candidates)
    if beam_scores is None:
        beam_scores = [0.0] * len(candidates)

    cand_beam: Dict[str, float] = {}
    for text, bs in zip(candidates, beam_scores):
        key = text.strip()
        if key and (key not in cand_beam or bs > cand_beam[key]):
            cand_beam[key] = bs

    bs_values = list(cand_beam.values())
    bs_min = min(bs_values) if bs_values else 0.0
    bs_max = max(bs_values) if bs_values else 0.0
    bs_range = bs_max - bs_min if bs_max > bs_min else 1.0

    avg_len = np.mean([len(c.split()) for c in unique])
    best_score = -1e18
    best_candidate = unique[0]

    for cand in unique:
        refs = [r for r in unique if r != cand]
        bleu_proxy = sentence_bleu_proxy(cand, refs)
        chrf_proxy = sentence_chrf_proxy(cand, refs)
        len_diff = abs(len(cand.split()) - avg_len)
        len_score = max(0.0, 100.0 - 2.0 * len_diff)

        vote_score = 0.0
        for pred, w in zip(candidates, model_weights):
            if pred.strip() == cand:
                vote_score += float(w)

        norm_beam = (cand_beam.get(cand, bs_min) - bs_min) / bs_range * 100.0

        final = (
            bleu_weight * bleu_proxy
            + chrf_weight * chrf_proxy
            + length_weight * len_score
            + beam_score_weight * norm_beam
            + 5.0 * vote_score
        )
        if final > best_score:
            best_score = final
            best_candidate = cand
    return best_candidate


def validate_submission_df(
    submission_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = "id",
    target_col: str = "translation",
) -> None:
    required = [id_col, target_col]
    missing = [c for c in required if c not in submission_df.columns]
    if missing:
        raise ValueError(f"Missing submission columns: {missing}")
    if len(submission_df) != len(test_df):
        raise ValueError(
            f"Submission row count {len(submission_df)} != test row count {len(test_df)}"
        )
    if submission_df[id_col].astype(str).tolist() != test_df[id_col].astype(str).tolist():
        raise ValueError("Submission id column does not match test id ordering.")
    if submission_df[target_col].isnull().any():
        raise ValueError("Submission contains null translations.")


def run_preprocessing_fixture() -> Dict[str, str]:
    sample = "szub x x … [x] {e₂} ḫa-lum 1''"
    normalized = canonicalize_text(sample, is_translation=False)
    return {"input": sample, "normalized": normalized}


def cli_self_check() -> None:
    metric = run_metric_fixture()
    prep = run_preprocessing_fixture()
    print("Metric fixture:", json.dumps(metric, indent=2))
    print("Preprocessing fixture:", json.dumps(prep, indent=2, ensure_ascii=True))
    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility self-checks")
    parser.add_argument("--self-check", action="store_true", help="Run utility fixtures")
    args = parser.parse_args()
    if args.self_check:
        cli_self_check()
