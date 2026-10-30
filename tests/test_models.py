import json

import pytest

from support_ticket.models.artifacts import load_artifact
from support_ticket.models.predict import Predictor
from support_ticket.models.train import build_model


def test_artifact_and_evaluation(artifact):
    bundle, metadata = load_artifact(artifact)
    assert bundle["version"] == metadata["model_version"]
    assert set(metadata["tasks"]["category"]["validation"]) == {"majority", "logistic", "svm"}
    metrics = metadata["tasks"]["category"]["test"]
    assert 0 <= metrics["macro_f1"] <= 1
    assert len(metrics["confusion_matrix"]) == 3
    assert metadata["synthetic"]


def test_predictions(artifact):
    predictor = Predictor(artifact)
    texts = ["invoice charge please", "password login not working"]
    outputs = predictor.predict(texts)
    assert len(outputs) == 2
    assert outputs[0] == predictor.predict(texts[:1])[0]
    assert outputs[0]["priority_status"] == "unavailable"
    assert all(r["category"]["review_required"] for r in outputs)
    for invalid in [[], [None], ["  "], ["x" * 10001], ["x"] * 101]:
        with pytest.raises(ValueError):
            predictor.predict(invalid)


def test_review_threshold(artifact):
    predictor = Predictor(artifact)
    predictor.metadata["synthetic"] = False
    predictor.config["always_review"] = False
    predictor.config["review_threshold"] = 1.0
    assert predictor.predict(["invoice payment"])[0]["category"]["review_required"]
    predictor.config["review_threshold"] = 0.0
    assert not predictor.predict(["invoice payment"])[0]["category"]["review_required"]


def test_svm_scores_and_no_feature_leakage(config):
    model = build_model("svm", config)
    model.fit(
        ["invoice payment", "password login", "invoice bill", "password reset"],
        ["Billing", "Access", "Billing", "Access"],
    )
    assert not hasattr(model, "predict_proba")
    vocab = model.named_steps["features"].transformer_list[0][1].vocabulary_
    model.predict(["uniquetestonlytoken"])
    assert "uniquetestonlytoken" not in vocab


def test_artifact_missing_and_mismatch(artifact, tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path)
    metadata = json.loads((artifact / "metadata.json").read_text())
    metadata["model_version"] = "wrong"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "pipeline.joblib").write_bytes((artifact / "pipeline.joblib").read_bytes())
    with pytest.raises(ValueError, match="versions"):
        Predictor(tmp_path)


def test_priority_training(config, frame, tmp_path):
    from support_ticket.data.splitting import split_data
    from support_ticket.data.validation import validate_data
    from support_ticket.models.train import train

    # Synthetic labels exercise wiring only; do not claim real priority suitability.
    config.priority_enabled = True
    config.priority_labels_verified = True
    config.columns["priority"] = "priority"
    frame["priority"] = frame.category.map(
        {"Billing": "Low", "Account Access": "High", "Technical Issue": "Medium"}
    )
    clean, quality = validate_data(frame, config)
    assert quality["valid"]
    metadata = train(split_data(clean, priority_enabled=True), config, tmp_path)
    assert "priority" in metadata["tasks"]
    result = Predictor(tmp_path / "artifact").predict(["login password"])[0]
    assert result["priority_status"] == "available"
    assert result["priority"]["review_required"]
