"""Noninteractive figures and content-free split manifests."""

import hashlib
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_quality(frame: pd.DataFrame, report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "quality.json").write_text(json.dumps(report, indent=2))
    for task in ["category", "priority"]:
        if task in frame:
            ax = frame[task].value_counts().plot.bar(title=f"{task} distribution")
            ax.set_ylabel("Tickets")
            plt.tight_layout()
            plt.savefig(out / f"{task}-distribution.png")
            plt.close()
    lengths = frame.ticket_text.map(lambda x: len(x) if isinstance(x, str) else 0)
    lengths.hist(bins=30)
    plt.title("Text length distribution")
    plt.xlabel("Characters")
    plt.ylabel("Tickets")
    plt.tight_layout()
    plt.savefig(out / "text-lengths.png")
    plt.close()


def write_manifest(splits: dict, out: Path, seed: int) -> None:
    manifest = {
        "seed": seed,
        "strategy": "60/20/20 category-stratified normalized-text groups",
        "splits": {},
    }
    for name, df in splits.items():
        manifest["splits"][name] = {
            "count": len(df),
            "groups": sorted(df.group_id.unique()),
            "record_id_hashes": [hashlib.sha256(t.encode()).hexdigest() for t in df.ticket_id],
            "category_counts": df.category.value_counts().to_dict(),
        }
    (out / "splits.json").write_text(json.dumps(manifest, indent=2))
