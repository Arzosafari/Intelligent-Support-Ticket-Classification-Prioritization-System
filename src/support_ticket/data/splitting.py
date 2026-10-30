"""Stratify unique normalized-text groups, then expand without dropping duplicates."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame, seed: int = 42, priority_enabled: bool = False
) -> dict[str, pd.DataFrame]:
    groups = df.drop_duplicates("group_id")
    counts = groups.category.value_counts()
    if len(counts) < 2 or (counts < 5).any():
        raise ValueError("Need two classes and five independent groups per class")
    try:
        train, remaining = train_test_split(
            groups, test_size=0.4, random_state=seed, stratify=groups.category
        )
        valid, test = train_test_split(
            remaining, test_size=0.5, random_state=seed, stratify=remaining.category
        )
    except ValueError as exc:
        raise ValueError("Insufficient independent groups for stratified 60/20/20 splits") from exc
    splits = {
        k: df[df.group_id.isin(v.group_id)].copy()
        for k, v in [("train", train), ("validation", valid), ("test", test)]
    }
    for task in ["category"] + (["priority"] if priority_enabled else []):
        expected = set(df[task])
        for name, part in splits.items():
            if set(part[task]) != expected:
                raise ValueError(
                    f"{name} lacks {task} class coverage; collect data or change strategy"
                )
    return splits
