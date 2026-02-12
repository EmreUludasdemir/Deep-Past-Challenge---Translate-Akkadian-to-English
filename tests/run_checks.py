import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import (
    canonicalize_text,
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


def main() -> None:
    test_metric_fixture()
    test_preprocessing_cases()
    test_submission_validation()
    print("All tests passed.")


if __name__ == "__main__":
    main()
