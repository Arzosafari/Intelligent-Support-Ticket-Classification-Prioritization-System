"""Trusted-local joblib persistence. NEVER deserialize untrusted artifacts."""

import json
from pathlib import Path

import joblib


def save_artifact(directory: Path, bundle: dict, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, directory / "pipeline.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_artifact(directory: str | Path) -> tuple[dict, dict]:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    bundle = joblib.load(directory / "pipeline.joblib")
    if bundle.get("version") != metadata.get("model_version"):
        raise ValueError("Artifact and metadata versions do not match")
    return bundle, metadata
