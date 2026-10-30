"""Fail closed on invalid training records; report issues without message contents."""

import hashlib

import pandas as pd

from support_ticket.config import Settings
from support_ticket.features.text import normalize_text


def validate_data(frame: pd.DataFrame, config: Settings) -> tuple[pd.DataFrame, dict]:
    required = {"ticket_id", "ticket_text", "category"}
    if config.priority_enabled:
        required.add("priority")
    if not required <= set(frame.columns):
        raise ValueError(f"Missing canonical fields: {sorted(required - set(frame.columns))}")
    df = frame.copy()
    report = {
        "records": len(df),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing": {
            k: {"count": int(v), "percent": float(v / max(len(df), 1) * 100)}
            for k, v in df.isna().sum().items()
        },
        "duplicate_records": int(df.duplicated().sum()),
        "issues": [],
    }
    issues = report["issues"]
    if df.empty:
        issues.append("Dataset is empty")
    invalid = {}
    report["problem_positions"] = {}
    for key in required:
        bad = ~df[key].map(lambda v: isinstance(v, str) and bool(v.strip()))
        invalid[key] = int(bad.sum())
        report["problem_positions"][key] = [i for i, value in enumerate(bad) if value][:20]
        if bad.any():
            issues.append(f"{key}: {int(bad.sum())} missing, empty or non-string values")
    report["invalid_counts"] = invalid
    long = df.ticket_text.map(lambda v: isinstance(v, str) and len(v) > config.max_text_length)
    if long.any():
        issues.append(f"{int(long.sum())} text values exceed configured maximum")
    if any(invalid.values()):
        report["valid"] = False
        return df, report
    for task, mapping, allowed in [
        ("category", config.category_mapping, config.categories),
        ("priority", config.priority_mapping, config.priorities),
    ]:
        if task not in df:
            continue
        df[task] = df[task].map(
            lambda x: mapping.get(x.strip(), x.strip()) if isinstance(x, str) else x
        )
        if df[task].map(lambda v: isinstance(v, str) and not v.strip()).any():
            issues.append(f"{task}: mapping produces empty labels")
        if allowed is not None and not set(df[task].dropna()) <= set(allowed):
            issues.append(f"{task}: unknown labels outside configured allowlist")
        report[f"{task}_distribution"] = df[task].value_counts().to_dict()
    df["group_id"] = df.ticket_text.map(
        lambda t: hashlib.sha256(normalize_text(t, config.mask_pii, True).encode()).hexdigest()
    )
    report["duplicate_text_records"] = int(df.ticket_text.duplicated().sum())
    report["normalized_duplicate_records"] = int(df.group_id.duplicated().sum())
    report["duplicate_text_groups"] = int((df.groupby("group_id").size() > 1).sum())
    report["duplicate_ticket_ids"] = int(df.ticket_id.duplicated().sum())
    if df.ticket_id.duplicated().any():
        issues.append("Repeated ticket_id; reconcile records before training")
    for task in ["category"] + (["priority"] if config.priority_enabled else []):
        conflicts = int((df.groupby("group_id")[task].nunique() > 1).sum())
        report[f"{task}_conflicting_groups"] = conflicts
        if conflicts:
            issues.append(f"{task}: {conflicts} normalized text groups have conflicting labels")
        groups = df.drop_duplicates("group_id")[task].value_counts()
        if len(groups) < 2 or (groups < 5).any():
            issues.append(
                f"{task}: need at least two classes and five independent groups per class"
            )
    report["text_length"] = df.ticket_text.str.len().describe().to_dict()
    report["warnings"] = [
        "Near-duplicate templates are not automatically detected; audit before real experiments.",
        "PII masking is heuristic, not a privacy guarantee.",
    ]
    report["valid"] = not issues
    return df, report
