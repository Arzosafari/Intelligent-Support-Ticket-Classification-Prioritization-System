# Operations and release runbook

## Startup and troubleshooting

- Liveness 200 with readiness 503 means the process runs without a usable model. Confirm `MODEL_DIR` points to a trusted directory containing both `pipeline.joblib` and `metadata.json`, readable by the service user. Compare package versions to training metadata. Retrain rather than loading arbitrary third-party pickle files. Deserialization failures are intentionally not returned to clients.
- 422: inspect the schema, content type, blank text, length limit, and batch bounds; error responses intentionally omit input. 413: upstream/client request is too large; split batches. 500: investigate internal diagnostics in a restricted environment without enabling raw-input logging.
- Quality rejection: read `quality.json`, reconcile IDs/conflicts, inspect source data privately, correct upstream or configure semantically defensible mappings. Never delete rare classes just to obtain a better score.
- Split rejection: collect independent examples, audit duplicate grouping and class coverage. For temporal/customer deployment scenarios implement and validate an explicit alternative splitter rather than weakening checks.
- Deployment is restart-based: stop old process and start with the approved artifact directory. A loaded in-memory model remains ready if its files later disappear. No hot reload or remote model download is performed.

## Monitoring design

The checked-in logging JSON suppresses Uvicorn access logging to avoid paths/query content. `http_request` JSON events contain generated correlation ID, HTTP method, status and latency, not user-provided IDs or text. `prediction_batch` events contain ticket count, category labels, model version, and review count. Labels must be a safe operational taxonomy. Request logs and prediction events are separate; correlation across these events is a future enhancement.

Send stdout to an approved aggregator with bounded retention (suggested initial policy: 14 days, reviewed by privacy/security). Aggregate:

| Signal | Derivation / action |
|---|---|
| Request and ticket volume | Count request events; sum batch count separately |
| Error / invalid rate | Status 5xx / 4xx, especially 422/413; alert on sustained increases |
| Latency | Percentiles of request `latency_ms`; split infrastructure vs inference in load studies |
| Version | Partition all prediction statistics by `model_version` |
| Prediction distribution | Histogram category labels; investigate abrupt change relative to an approved reference window |
| Review rate | Sum review counts / predicted ticket counts; high rates may be intentional (default 100%) |
| Ground-truth performance | Join approved independently reviewed labels with versioned predictions, then compute delayed class recall and macro F1 |
| Drift | Distribution shifts are signals only; add consented aggregate length/language features in a future release |

No ground-truth collection exists in the API, so **live accuracy is not available**. Distribution drift does not prove accuracy degradation. Avoid arbitrary universal alerts: establish traffic baselines and business-approved windows and thresholds first. No load/SLA validation was performed.

## Feedback and retraining

1. Collect only approved ticket references, prediction/version, and a human-confirmed label in a separately access-controlled feedback store. Do not turn model predictions into ground truth. Text access requires explicit retention authorization.
2. Deduplicate with historical training groups; audit taxonomy consistency, disagreement, rare classes and privacy. Keep feedback out of training until validated. Record license/provenance and source fingerprint.
3. Freeze a new temporal or group holdout appropriate to deployment; use training/validation for experiments. Retire an old test set to development only by explicitly creating a fresh untouched test set.
4. Train candidates with versioned config/dependencies. Inspect class recall, confusion pairs, review coverage, validation calibration, latency, memory and domain slices. Set release acceptance criteria before opening the test results.
5. Evaluate the frozen candidate once on test. Independently review results, artifact checksum, training environment, security scans and model card. Approve deployment, don't auto-promote based on one metric.
6. Stage with read-only artifacts, smoke-test `/ready` and synthetic inference, then canary under human review. Preserve old artifact + config + environment for rollback.
7. Roll back by restarting with the last approved artifact and its compatible environment; compare version and monitoring. Never overwrite an experiment directory or mutate a running model.

## Container and infrastructure gates

Local Docker cannot be tested in this runtime. CI is configured to build and test it, not evidence it has passed. Before rollout verify image architecture/digest, read-only mounts, health checks, process shutdown, quotas, timeouts, request concurrency, trusted proxy routing, TLS, identity/authorization, rate limits, secret injection and log redaction. Evaluate dependency vulnerabilities and remove dev dependencies if desired after validating an independently locked runtime image.
