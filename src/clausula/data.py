"""Load the LexGLUE `unfair_tos` dataset as pandas DataFrames."""

import pandas as pd
from datasets import load_dataset

from clausula.labels import LABELS


def load_splits() -> dict[str, pd.DataFrame]:
    """Return {"train", "validation", "test"} DataFrames.

    Columns: `text`, `labels` (list of label ids), one 0/1 column per label
    (multi-hot encoding) and `is_unfair` (1 if the sentence has any label).
    """
    ds = load_dataset("coastalcph/lex_glue", "unfair_tos")
    splits = {}
    for name in ("train", "validation", "test"):
        df = ds[name].to_pandas()
        df["text"] = df["text"].str.strip()
        for i, label in enumerate(LABELS):
            df[label] = df["labels"].apply(lambda ids, i=i: int(i in ids))
        df["is_unfair"] = (df[LABELS].sum(axis=1) > 0).astype(int)
        splits[name] = df
    return splits
