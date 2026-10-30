import pandas as pd
import pytest

from support_ticket.config import Settings
from support_ticket.data.ingestion import load_data
from support_ticket.data.splitting import split_data
from support_ticket.data.validation import validate_data
from support_ticket.features.text import TextNormalizer, normalize_text


@pytest.mark.parametrize("value", [None, 42, "", " \n\t", float("nan")])
def test_invalid_text(value):
    with pytest.raises(ValueError):
        normalize_text(value)


def test_normalization():
    assert normalize_text(" Ｅrror  v1.2\n not working E403 ") == "Error v1.2 not working E403"
    assert "alice@example.org" not in normalize_text("write alice@example.org")
    assert normalize_text("A@b.org", mask_pii=False) == "A@b.org"
    assert TextNormalizer().fit_transform([" help  now "]) == ["help now"]


@pytest.mark.parametrize(
    "column,value",
    [("ticket_text", ""), ("ticket_text", None), ("category", None), ("ticket_id", 2)],
)
def test_invalid_data(config, frame, column, value):
    frame.loc[0, column] = value
    _, report = validate_data(frame, config)
    assert not report["valid"]


def test_unknown_label(config, frame):
    config.categories = ["Billing"]
    _, report = validate_data(frame, config)
    assert not report["valid"]


def test_conflicts_and_ids(config, frame):
    frame.loc[1, "ticket_text"] = frame.loc[0, "ticket_text"]
    frame.loc[1, "category"] = "different"
    frame.loc[1, "ticket_id"] = frame.loc[0, "ticket_id"]
    _, report = validate_data(frame, config)
    assert report["category_conflicting_groups"] == 1
    assert report["duplicate_ticket_ids"] == 1
    assert not report["valid"]


def test_splits(config, frame):
    duplicate = frame.iloc[[0]].copy()
    duplicate["ticket_id"] = "new-independent-ticket"
    frame = pd.concat([frame, duplicate], ignore_index=True)
    clean, quality = validate_data(frame, config)
    assert quality["valid"]
    assert quality["duplicate_text_records"] == 1
    splits = split_data(clean)
    assert sum(map(len, splits.values())) == len(frame)
    sets = [set(s.group_id) for s in splits.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert split_data(clean)["test"].ticket_id.tolist() == splits["test"].ticket_id.tolist()
    assert all(set(p.category) == set(frame.category) for p in splits.values())


def test_rare_and_schema(config, frame):
    _, report = validate_data(frame.head(3), config)
    assert not report["valid"]
    with pytest.raises(ValueError, match="Missing canonical"):
        validate_data(frame.drop(columns="category"), config)
    with pytest.raises(ValueError):
        split_data(validate_data(frame.head(3), config)[0])


def test_ingestion(tmp_path, frame, config):
    path = tmp_path / "tickets.csv"
    frame.rename(columns={"ticket_text": "message"}).to_csv(path, index=False)
    config.columns["ticket_text"] = "message"
    assert len(load_data(path, config)) == len(frame)
    config.columns["ticket_text"] = "missing"
    with pytest.raises(ValueError, match="Missing mapped"):
        load_data(path, config)
    with pytest.raises(ValueError, match="Supported"):
        load_data(tmp_path / "data.exe", config)


def test_jsonl_types(tmp_path, config):
    path = tmp_path / "data.jsonl"
    path.write_text('{"ticket_id":"one","ticket_text":123,"category":"Billing"}\n')
    _, report = validate_data(load_data(path, config), config)
    assert not report["valid"]


def test_priority_requires_verification(config):
    values = config.model_dump()
    values["priority_enabled"] = True
    with pytest.raises(ValueError, match="verification"):
        Settings(**values)


def test_empty_mapped_labels(config, frame):
    config.category_mapping = {"Billing": ""}
    _, report = validate_data(frame, config)
    assert not report["valid"]
    assert any("empty labels" in issue for issue in report["issues"])
