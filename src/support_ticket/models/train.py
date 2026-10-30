"""Fit candidates on train, select on validation, evaluate winner once on test."""

import hashlib
import importlib.metadata
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from support_ticket.config import Settings
from support_ticket.features.text import TextNormalizer
from support_ticket.models.artifacts import save_artifact
from support_ticket.models.evaluate import error_analysis, evaluate


def build_model(name: str, config: Settings) -> Pipeline:
    classifiers = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1500, random_state=config.seed),
        "svm": LinearSVC(random_state=config.seed),
    }
    vectors = [
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, config.word_ngram_max),
                min_df=config.min_df,
                max_features=config.max_features,
                lowercase=False,
            ),
        )
    ]
    if config.character_ngrams:
        vectors.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=config.min_df,
                    max_features=config.max_features,
                    lowercase=False,
                ),
            )
        )
    return Pipeline(
        [
            ("normalize", TextNormalizer(config.mask_pii, config.lowercase)),
            ("features", FeatureUnion(vectors)),
            ("classifier", classifiers[name]),
        ]
    )


def train(splits: dict, config: Settings, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    canonical = splits["train"].columns.tolist()
    fingerprint = hashlib.sha256(
        "\n".join(
            splits[k][canonical].sort_values("ticket_id").to_json(orient="records")
            for k in sorted(splits)
        ).encode()
    ).hexdigest()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + fingerprint[:8]
    metadata = {
        "model_version": version,
        "dataset_id": config.dataset_id,
        "dataset_fingerprint": fingerprint,
        "synthetic": config.synthetic,
        "source_license": config.source_license,
        "config": config.model_dump(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "software": {
            p: importlib.metadata.version(p)
            for p in ["scikit-learn", "numpy", "scipy", "pandas", "joblib"]
        },
        "selection_policy": "highest validation macro F1; ties prefer majority then logistic",
        "limitations": [
            "No calibrated probabilities",
            "No OOD detector",
            "Synthetic results do not estimate real-world performance",
            "Near-duplicate templates require manual audit",
        ],
        "tasks": {},
    }
    bundle = {"version": version, "models": {}}
    for task in ["category"] + (["priority"] if config.priority_enabled else []):
        classes = sorted(splits["train"][task].unique())
        fitted, metrics = {}, {}
        for name in ["majority", "logistic", "svm"]:
            logging.getLogger(__name__).info("training task=%s candidate=%s", task, name)
            model = build_model(name, config)
            model.fit(splits["train"].ticket_text, splits["train"][task])
            fitted[name] = model
            metrics[name] = evaluate(
                model, splits["validation"].ticket_text, splits["validation"][task], classes
            )
        selected = max(metrics, key=lambda n: metrics[n]["macro_f1"])
        winner = fitted[selected]
        details = {
            "selected": selected,
            "labels": classes,
            "validation": metrics,
            "test": evaluate(winner, splits["test"].ticket_text, splits["test"][task], classes),
            "parameters": winner.named_steps["classifier"].get_params(),
            "feature_configuration": {
                name: {
                    key: value
                    for key, value in vector.get_params().items()
                    if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
                }
                for name, vector in winner.named_steps["features"].transformer_list
            },
            "training_distribution": splits["train"][task].value_counts().to_dict(),
        }
        (out / f"{task}-errors.json").write_text(
            json.dumps(error_analysis(winner, splits["validation"], task), indent=2)
        )
        metadata["tasks"][task] = details
        bundle["models"][task] = winner
    save_artifact(out / "artifact", bundle, metadata)
    (out / "evaluation.json").write_text(json.dumps(metadata, indent=2))
    return metadata
