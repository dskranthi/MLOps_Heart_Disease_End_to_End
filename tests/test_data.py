import pandas as pd

from src.data import CATEGORICAL, COLUMNS, FEATURES, NUMERIC, binarize_target, clean, load_raw


def test_schema_constants():
    assert len(FEATURES) == 13 and len(COLUMNS) == 14
    assert set(CATEGORICAL) | set(NUMERIC) == set(FEATURES)
    assert not set(CATEGORICAL) & set(NUMERIC)


def test_binarize_target():
    out = binarize_target(pd.Series([0, 1, 2, 3, 4]))
    assert out.tolist() == [0, 1, 1, 1, 1]


def test_clean_binary_target_and_keeps_nans(synthetic_raw):
    out = clean(synthetic_raw)
    assert set(out["target"].unique()) <= {0, 1}
    assert out["ca"].isna().sum() >= 1  # imputed later, inside the pipeline


def test_clean_drops_duplicates_and_missing_target(synthetic_raw):
    dup = pd.concat([synthetic_raw, synthetic_raw.iloc[[0]]], ignore_index=True)
    dup.loc[1, "target"] = None
    out = clean(dup)
    assert len(out) <= len(synthetic_raw)
    assert out["target"].notna().all()


def test_load_raw_parses_question_mark(tmp_path):
    p = tmp_path / "x.data"
    p.write_text("63,1,1,145,233,1,2,150,0,2.3,3,?,6,0\n")
    df = load_raw(p)
    assert df.shape == (1, 14) and pd.isna(df.loc[0, "ca"])
