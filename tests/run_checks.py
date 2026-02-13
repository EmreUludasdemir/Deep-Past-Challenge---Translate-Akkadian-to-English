import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import (
    canonicalize_text,
    consensus_rerank,
    mine_dictionary_pairs,
    repair_named_entities,
    run_metric_fixture,
    validate_submission_df,
)


def test_metric_fixture() -> None:
    result = run_metric_fixture()
    assert result["geometric_mean"] > 0.0


def test_preprocessing_cases() -> None:
    cases = {
        "szu-ba-at x x ...": "<big_gap>",
        "{e₂} 1'' KISZIB": "{e2}",
        "s, t, sz": "ṣ ṭ š",
        "x [x] [x x]": "<big_gap>",
    }
    for raw, required in cases.items():
        out = canonicalize_text(raw, is_translation=False)
        assert required in out, f"expected {required} in {out}"


def test_submission_validation() -> None:
    test_df = pd.DataFrame({"id": [0, 1], "transliteration": ["a", "b"]})
    sub_df = pd.DataFrame(
        {"id": [0, 1], "translation": ["Send a letter.", "Bring silver."]}
    )
    validate_submission_df(sub_df, test_df, id_col="id", target_col="translation")


def test_task_prefix_prepending() -> None:
    prefix = "translate Akkadian to English: "
    source = "a-na A-szur"
    result = prefix + source
    assert result.startswith(prefix)
    assert result.endswith(source)
    assert len(result) == len(prefix) + len(source)


def test_named_entity_repair_multiple() -> None:
    source = "Aszur-idi a-na Pusu-ken"
    lexicon = {"aszur-idi", "pusu-ken"}
    prediction = "He said to the man."
    repaired = repair_named_entities(prediction, source, lexicon=lexicon, max_append=2)
    lower = repaired.lower()
    assert "aszur-idi" in lower or "pusu-ken" in lower, f"Expected entity in: {repaired}"


def test_named_entity_repair_present() -> None:
    source = "Aszur-idi a-na Pusu-ken"
    lexicon = {"aszur-idi", "pusu-ken"}
    prediction = "Aszur-idi said to Pusu-ken."
    repaired = repair_named_entities(prediction, source, lexicon=lexicon, max_append=2)
    assert repaired == prediction, f"Should not modify: {repaired}"


def test_dictionary_mining_empty() -> None:
    result = mine_dictionary_pairs(None)
    assert result.empty
    assert list(result.columns) == ["source", "target", "group_id", "origin"]


def test_dictionary_mining_with_data() -> None:
    df = pd.DataFrame({
        "form": ["a-lum", "be-lum", ""],
        "meaning": ["city", "lord", ""],
    })
    result = mine_dictionary_pairs(df, max_pairs=100)
    assert len(result) == 2
    assert "origin" in result.columns
    assert all(result["origin"] == "dictionary")


def test_consensus_rerank_with_beam_scores() -> None:
    candidates = ["hello world", "hello there", "hello world"]
    weights = [0.5, 0.3, 0.2]
    beam_scores = [-0.5, -1.2, -0.6]
    result = consensus_rerank(
        candidates=candidates,
        model_weights=weights,
        beam_scores=beam_scores,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def main() -> None:
    test_metric_fixture()
    test_preprocessing_cases()
    test_submission_validation()
    test_task_prefix_prepending()
    test_named_entity_repair_multiple()
    test_named_entity_repair_present()
    test_dictionary_mining_empty()
    test_dictionary_mining_with_data()
    test_consensus_rerank_with_beam_scores()
    print("All tests passed.")


if __name__ == "__main__":
    main()
