# Implementation and verification record

Date: 2026-10-24 (UTC). Repository began with only its original `LICENSE`, clean Git status, no dataset/dependencies/tests. Active session branch was preserved; no commits, push, PR, deletion or modification of existing license was performed. Python 3.11.2, CPU/Linux x86-64. Docker executable absent. System pip refused installation under PEP 668; all subsequent work used `.venv` without overriding system protections.

## Milestones and dependencies

| Milestone | Implementation / evidence | Status |
|---|---|---|
| 1 Inspection, assumptions, architecture | README, strict config, diagrams, documented unverified candidate source | Complete for available inputs; real data/license pending |
| 2 Data, EDA, splits | CSV/JSONL mapping, fail-closed quality reports, PNGs, group manifests; unit tests and executed CLI | Verified on synthetic data |
| 3 Majority/LR/SVM and metrics | Fixed train/validation/test, validation selection, selected-model test report | Verified on synthetic data only |
| 4 Errors/comparison/priority | Validation error JSON, candidate metrics; priority disabled by default; synthetic optional wiring test | Real errors and priority suitability pending real labels |
| 5 Artifacts/inference | Complete joblib pipeline, versioned metadata, save/load tests, single/batch parity | Local trusted artifact verified |
| 6 API/tests | Strict schemas, readiness probe, safe metadata/errors, body caps, structured events, HTTP integration | Verified locally |
| 7 CI/Docker/operations | CI workflow, non-root Dockerfile, allowlist context, runbook, pinned environment | Configuration complete; Docker/GitHub execution unverified |
| 8 Integration/presentation | 38 passing tests, lint/format, CLI workflow, wheel and live HTTP probes, documentation | Local synthetic integration verified; production validation pending |

## Commands actually executed

After `.venv` creation and dependency installation:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -q
.venv/bin/pip check
.venv/bin/python -m support_ticket.cli make-fixture
.venv/bin/python -m support_ticket.cli validate --data data/raw/synthetic.csv --out reports/runs/validation
.venv/bin/python -m support_ticket.cli split --data data/raw/synthetic.csv --out reports/runs/split
.venv/bin/python -m support_ticket.cli train --data data/raw/synthetic.csv --out reports/runs/demo
.venv/bin/python -m support_ticket.cli predict --artifact reports/runs/demo/artifact --text 'I cannot login to my account'
.venv/bin/python -m support_ticket.cli train --data data/raw/synthetic.csv --out reports/runs/final
.venv/bin/pip wheel --no-deps --no-build-isolation . -w /home/user/wheels
```

`reports/runs/final/` supplies the checked-in synthetic sample reports. `reports/runs/`, raw fixture and artifacts are ignored, so a fresh checkout must train locally. Generated binaries/wheels are not part of the patch. README quickstart uses a fresh demo directory; choose new run paths if repeating commands in this workspace.

### Results

- Ruff: all checks passed; 24 Python files formatted.
- Pytest final run: **38 passed, 1 warning**, about 9 seconds. Warning: Starlette TestClient uses an AnyIO deprecated alias; no warnings suppressed.
- `pip check`: no broken requirements.
- CLI fixture/validation/split/train/predict: exit 0. Tests additionally verified data rejection exit 2 and refusal to overwrite a nonempty report directory.
- Wheel build: success; wheel contains the runtime logging JSON.
- Initial tests caught an incorrect shallow Pipeline parameter lookup and insufficient independent-class guard in splitting. Fixed both, then reran the suite. Empty mapped labels also now reject.
- Latest executed synthetic artifact version and timings are in `reports/synthetic-evaluation.json`; fingerprint is stable between demo and final runs. Test is 18 fixture rows, with 1.0 macro F1. **Not a real-world evaluation.**
- Startup via managed background Uvicorn on `0.0.0.0:8000`: success. Real HTTP probes verified `/health`, `/ready`, `/model-info`, `/docs`, `/predict`, `/predict/batch` all 200; IDs retained in order; blank text 422; root redirects to `/docs` (307). Structured logs contain counts/categories/version/status/latency, not ticket content. TestClient separately verified body 413, missing/corrupt model 503, malformed/type/batch errors 422.
- Git diff whitespace checks: clean. Original license unchanged.

## Not executed / not established

- Real data acquisition, license verification, label semantics review, real-world baseline performance, actual priority performance, calibration/OOD/fairness studies.
- Container build or startup: Docker is not installed here. GitHub Actions execution, cloud deployment, external identity/TLS/rate-limit integration.
- Load, memory, concurrency, p95/p99 latency or SLA verification. Reported latency is a single warm batch, not an API benchmark.
- Public deployment security, true anonymization, signed artifacts, automatic near-duplicate/template grouping, temporal/customer splits, production drift monitoring, or a feedback persistence service.

## Next external actions

1. Verify a suitable source's license/provenance and taxonomy; configure its real columns and reviewed labels. Keep private content out of Git/logs.
2. Audit leakage groups and split strategy, run genuine experiments, inspect class-level failures and human-review economics. Reserve a fresh untouched test set for each genuinely new evaluation regime.
3. Run Docker build/startup smoke tests on a Docker-capable host; inspect actual CI results after an authorized push. Do not infer they passed from configuration files.
4. Approve calibration/review policy and deployment security controls before exposure. Start with mandatory human review and retain rollback artifacts.
