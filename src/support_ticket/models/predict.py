"""Framework-independent bounded inference with explicitly uncalibrated scores."""

from pathlib import Path

import numpy as np

from support_ticket.features.text import normalize_text
from support_ticket.models.artifacts import load_artifact


class Predictor:
    """Load only a trusted local artifact; no customer messages are persisted."""

    def __init__(self, directory: str | Path):
        self.bundle, self.metadata = load_artifact(directory)
        self.config = self.metadata["config"]
        # Fail startup if an artifact is present but unusable.
        self.predict(["artifact readiness probe"])

    def predict(self, texts: list[str]) -> list[dict]:
        if not isinstance(texts, list) or not 1 <= len(texts) <= 100:
            raise ValueError("Provide a list containing 1 to 100 tickets")
        for text in texts:
            normalize_text(text)
            if len(text) > self.config["max_text_length"]:
                raise ValueError("Ticket exceeds model text length limit")
        results = [
            {
                "model_version": self.metadata["model_version"],
                "priority": None,
                "priority_status": "unavailable",
            }
            for _ in texts
        ]
        for task, model in self.bundle["models"].items():
            labels = model.predict(texts)
            scores = model.predict_proba(texts) if hasattr(model, "predict_proba") else None
            for i, label in enumerate(labels):
                score = float(np.max(scores[i])) if scores is not None else None
                review = (
                    self.config["always_review"]
                    or score is None
                    or score < self.config["review_threshold"]
                    or self.metadata["synthetic"]
                )
                results[i][task] = {
                    "label": str(label),
                    "score": score,
                    "score_type": "uncalibrated_probability"
                    if score is not None
                    else "unavailable",
                    "review_required": review,
                    "review_reason": "conservative_policy"
                    if review
                    else "above_operational_threshold",
                }
                if task == "priority":
                    results[i]["priority_status"] = "available"
        return results

    def safe_info(self) -> dict:
        """Allowlist metadata, excluding dataset paths and arbitrary config content."""
        return {
            "model_version": self.metadata["model_version"],
            "synthetic": self.metadata["synthetic"],
            "trained_at": self.metadata["trained_at"],
            "tasks": {
                k: {"selected": v["selected"], "labels": v["labels"]}
                for k, v in self.metadata["tasks"].items()
            },
            "limitations": self.metadata["limitations"],
        }
