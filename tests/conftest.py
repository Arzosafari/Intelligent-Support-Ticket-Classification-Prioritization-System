"""All tests use local, non-sensitive synthetic examples."""

import pytest

from support_ticket.config import load_settings
from support_ticket.data.splitting import split_data
from support_ticket.data.synthetic import synthetic_data
from support_ticket.data.validation import validate_data
from support_ticket.models.train import train


@pytest.fixture
def config():
    return load_settings("configs/config.example.yaml")


@pytest.fixture
def frame():
    return synthetic_data()


@pytest.fixture(scope="session")
def artifact(tmp_path_factory):
    config = load_settings("configs/config.example.yaml")
    frame, report = validate_data(synthetic_data(), config)
    assert report["valid"]
    out = tmp_path_factory.mktemp("experiment")
    train(split_data(frame, config.seed), config, out)
    return out / "artifact"
