"""Consistent label-aware metrics and content-free error analysis."""

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline


def evaluate(
    model: Pipeline,
    texts: Sequence[str] | pd.Series,
    labels: Sequence[str] | pd.Series,
    classes: list[str],
) -> dict:
    start = time.perf_counter()
    predicted = model.predict(texts)
    elapsed = time.perf_counter() - start
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=classes, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(
            precision_recall_fscore_support(labels, predicted, average="weighted", zero_division=0)[
                2
            ]
        ),
        "labels": classes,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=classes).tolist(),
        "classification_report": classification_report(
            labels, predicted, labels=classes, output_dict=True, zero_division=0
        ),
        "batch_ms_per_ticket": elapsed * 1000 / len(texts),
        "latency_method": "one warm-process batch; no SLA or cold-start claim",
    }


def error_analysis(model: Pipeline, frame: pd.DataFrame, task: str) -> dict:
    predictions = model.predict(frame.ticket_text)
    errors = frame.loc[np.asarray(predictions) != frame[task].to_numpy()].copy()
    errors["predicted"] = predictions[np.asarray(predictions) != frame[task].to_numpy()]
    pairs = errors.groupby([task, "predicted"]).size().sort_values(ascending=False)
    return {
        "source_split": "validation",
        "error_count": len(errors),
        "confusion_pairs": [
            {"actual": a, "predicted": b, "count": int(n)} for (a, b), n in pairs.items()
        ],
        "examples": [
            {
                "example": f"validation-error-{i}",
                "text": "[OMITTED FOR PRIVACY]",
                "length": len(row.ticket_text),
                "actual": row[task],
                "predicted": row.predicted,
            }
            for i, (_, row) in enumerate(errors.head(30).iterrows())
        ],
        "short_text_errors": int((errors.ticket_text.str.len() < 40).sum()),
        "per_class_errors": errors[task].value_counts().to_dict(),
        "next_experiments": [
            "Audit validation confusion pairs for ambiguous taxonomy and label errors.",
            "Inspect template groups and time/customer grouping before deployment.",
            "Collect rare-class and short-message labels; measure class recall changes.",
            "Evaluate calibration and review thresholds on separate validation labels.",
        ],
        "caution": "No automated label-error diagnosis; do not tune against final test errors.",
    }
