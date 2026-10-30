"""Strict, reproducible experiment configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    synthetic: bool = False
    source_license: str
    columns: dict[str, str] = Field(
        default_factory=lambda: {
            "ticket_id": "ticket_id",
            "ticket_text": "ticket_text",
            "category": "category",
        }
    )
    category_mapping: dict[str, str] = Field(default_factory=dict)
    categories: list[str] | None = None
    priority_enabled: bool = False
    priority_labels_verified: bool = False
    priority_mapping: dict[str, str] = Field(default_factory=dict)
    priorities: list[str] | None = None
    seed: int = 42
    mask_pii: bool = True
    lowercase: bool = False
    max_text_length: int = Field(default=10000, ge=100, le=100000)
    word_ngram_max: int = Field(default=2, ge=1, le=3)
    character_ngrams: bool = True
    min_df: int = Field(default=1, ge=1)
    max_features: int = Field(default=30000, ge=100)
    review_threshold: float = Field(default=0.7, ge=0, le=1)
    always_review: bool = True

    @model_validator(mode="after")
    def check_contract(self) -> "Settings":
        if not {"ticket_id", "ticket_text", "category"} <= self.columns.keys():
            raise ValueError("columns must map ticket_id, ticket_text and category")
        if len(set(self.columns.values())) != len(self.columns):
            raise ValueError("Source columns must be mapped only once")
        if self.priority_enabled and not self.priority_labels_verified:
            raise ValueError("Priority requires explicit verification of label suitability")
        if self.priority_enabled and "priority" not in self.columns:
            raise ValueError("Enabled priority requires a priority column mapping")
        return self


def load_settings(path: str | Path) -> Settings:
    """Load YAML without accepting executable constructors or unknown options."""
    return Settings.model_validate(yaml.safe_load(Path(path).read_text()))
