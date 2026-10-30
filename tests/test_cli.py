"""Exercise documented executable workflows and non-overwrite policy."""

import json
import subprocess
import sys


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "support_ticket.cli", *args], capture_output=True, text=True
    )


def test_cli_workflow(tmp_path):
    data, out = tmp_path / "data.csv", tmp_path / "run"
    assert run("make-fixture", "--output", str(data)).returncode == 0
    command = ("train", "--data", str(data), "--out", str(out))
    trained = run(*command)
    assert trained.returncode == 0, trained.stderr
    assert json.loads(trained.stdout)["synthetic"]
    predicted = run("predict", "--artifact", str(out / "artifact"), "--text", "invoice charge")
    assert predicted.returncode == 0, predicted.stderr
    assert json.loads(predicted.stdout)[0]["category"]["review_required"]
    assert run(*command).returncode == 2


def test_cli_rejected_data_report(tmp_path):
    data, out = tmp_path / "bad.csv", tmp_path / "quality"
    data.write_text("ticket_id,ticket_text,category\na,,Billing\n")
    assert run("validate", "--data", str(data), "--out", str(out)).returncode == 2
    assert not json.loads((out / "quality.json").read_text())["valid"]
