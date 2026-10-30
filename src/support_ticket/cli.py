"""Explicit local workflows; generated datasets/artifacts remain outside Git."""

import argparse
import json
import logging
from pathlib import Path

from support_ticket.config import load_settings
from support_ticket.data.ingestion import load_data
from support_ticket.data.reports import write_manifest, write_quality
from support_ticket.data.splitting import split_data
from support_ticket.data.synthetic import synthetic_data
from support_ticket.data.validation import validate_data
from support_ticket.models.predict import Predictor
from support_ticket.models.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("make-fixture", help="Write clearly synthetic test/demo CSV")
    fixture.add_argument("--output", default="data/raw/synthetic.csv")
    for command in ["validate", "split", "train"]:
        p = sub.add_parser(command)
        p.add_argument("--config", default="configs/config.example.yaml")
        p.add_argument("--data", required=True)
        p.add_argument("--out", default="reports/runs/demo")
    predict = sub.add_parser("predict")
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--text", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "make-fixture":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        synthetic_data().to_csv(output, index=False)
        print(f"Wrote SYNTHETIC fixture: {output}")
        return
    if args.command == "predict":
        print(json.dumps(Predictor(args.artifact).predict([args.text]), indent=2))
        return
    config = load_settings(args.config)
    df, quality = validate_data(load_data(args.data, config), config)
    out = Path(args.out)
    # Preserve historical reports rather than silently replacing an experiment.
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is not empty; choose a new run directory")
    write_quality(df, quality, out)
    if not quality["valid"]:
        parser.exit(2, f"Data rejected; inspect {out / 'quality.json'}\n")
    if args.command == "validate":
        print(f"Valid data; inspect {out / 'quality.json'}")
        return
    splits = split_data(df, config.seed, config.priority_enabled)
    write_manifest(splits, out, config.seed)
    if args.command == "train":
        result = train(splits, config, out)
        print(
            json.dumps(
                {
                    "synthetic": result["synthetic"],
                    "tasks": {
                        t: {"selected": v["selected"], "test_macro_f1": v["test"]["macro_f1"]}
                        for t, v in result["tasks"].items()
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
