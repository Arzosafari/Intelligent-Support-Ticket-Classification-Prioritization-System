"""Read local CSV/JSONL sources with explicitly mapped canonical fields."""

from pathlib import Path

import pandas as pd

from support_ticket.config import Settings


def load_data(path: str | Path, config: Settings) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix.lower() == ".jsonl":
        frame = pd.read_json(path, lines=True, dtype=False)
    else:
        raise ValueError("Supported formats: .csv and .jsonl")
    missing = set(config.columns.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing mapped columns: {sorted(missing)}")
    # Exclude unrelated source fields (potential PII and accidental features).
    return frame[list(config.columns.values())].rename(
        columns={source: target for target, source in config.columns.items()}
    )
