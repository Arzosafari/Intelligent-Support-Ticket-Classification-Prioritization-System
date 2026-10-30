"""Stateless normalization shared by ingestion, splitting and fitted pipelines."""

import re
import unicodedata
from collections.abc import Iterable

from sklearn.base import BaseEstimator, TransformerMixin


def normalize_text(text: str, mask_pii: bool = True, lowercase: bool = False) -> str:
    """Preserve negation, error codes and versions; mask common identifiers best-effort."""
    if not isinstance(text, str):
        raise ValueError("Ticket text must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    if not text:
        raise ValueError("Ticket text must not be empty")
    if mask_pii:
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " EMAIL ", text)
        text = re.sub(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)", " NUMBER ", text)
    return " ".join((text.lower() if lowercase else text).split())


class TextNormalizer(TransformerMixin, BaseEstimator):
    """Serializable sklearn transformer; no learned state or external downloads."""

    def __init__(self, mask_pii: bool = True, lowercase: bool = False):
        self.mask_pii = mask_pii
        self.lowercase = lowercase

    def fit(self, X: Iterable[str], y: Iterable[str] | None = None) -> "TextNormalizer":
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        return [normalize_text(x, self.mask_pii, self.lowercase) for x in X]
