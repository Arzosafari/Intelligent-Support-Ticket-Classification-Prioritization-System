# Engineering learning guide

Read alongside the executable modules and tests. A library call is not a substitute for understanding its assumptions. Each stage below covers purpose, necessity, concept, decisions, mistakes, verification, diagnosis, and an interview question.

## A. Data understanding (`data/ingestion.py`, `validation.py`, `reports.py`)

- **What:** map source fields and summarize missing/invalid values, types, duplicates, class counts and lengths; generate plots.
- **Why:** garbage or mislabeled data invalidates even a technically correct training loop.
- **Concept:** a data contract specifies the meaning and constraints of observations, not only column names.
- **Decisions:** local CSV/JSONL; explicit canonical mapping; no customer fields carried through accidentally; content-free reports. Dataset license is a manual governance gate, not something a parser can certify.
- **Common mistakes:** assuming taxonomy, converting missing text to the word `nan`, using post-resolution fields, conflating ticket status with urgency, redistributing restricted data.
- **Verify:** run `support-ticket validate`; inspect `quality.json` and plotted counts; tests inject bad schemas/types/labels.
- **When it fails:** inspect source parsing, mapping keys, invalid counts, allowlists and group conflicts privately. Report indices/content are not exported; use an authorized local audit to find records.
- **Interview:** “Why isn't high validation accuracy enough to establish label validity?” It measures agreement with supplied labels, not the correctness or business meaning of those labels.

## B. Splitting (`data/splitting.py`)

- **What:** stratify unique normalized-text groups into 60/20/20 train/validation/test, then expand back to rows.
- **Why:** duplicate leakage can make memorization look like generalization.
- **Concept:** independent evaluation units may be templates/customers/groups, not rows; stratification preserves marginal class representation, not independence.
- **Decisions:** seed 42, group hashes, retain duplicate independent tickets, fail for inadequate class coverage, no silent fallback to random row splits. Priority shares split assignments, but only category is stratified.
- **Common mistakes:** splitting before duplicate analysis, fitting vocabulary on all data, stratifying tiny classes, treating random split results as future-time performance. Near-template leakage remains possible here.
- **Verify:** inspect split hashes; test pairwise disjoint groups, class coverage, record counts and deterministic IDs. Hashes still require access control.
- **When it fails:** check independent group counts, conflicts, fraction rounding and optional priority coverage. For deployment on future tickets implement timestamp parsing and a documented time boundary.
- **Interview:** “Why not drop every repeated message?” Identical messages can be legitimate independent tickets; retaining them in one partition preserves their frequency without crossing the evaluation boundary.

## C. Text processing (`features/text.py`)

- **What:** validate strings, normalize NFKC/whitespace, optionally mask common emails/long numbers and lowercase.
- **Why:** the same policy must run at training and inference; useful technical tokens should survive.
- **Concept:** stateless normalization differs from learned vocabulary/IDF. A sklearn pipeline binds both to the classifier and prevents inconsistent inference.
- **Decisions:** no stop-word filtering or stemming, preserve negation/error codes/version numbers; best-effort PII masking is not full anonymization. Group keys always lowercase; model lowercasing is independently configurable.
- **Common mistakes:** deleting “not”, stripping digits from error codes, logging input in validation errors, assuming regex masking removes names/addresses/secrets.
- **Verify:** normalization unit tests include Unicode, whitespace, versions, missing values and PII examples; artifact reload reproduces predictions.
- **When it fails:** examine normalization policy and vocabulary on approved examples; inspect whether identifiers carried important or spurious signal.
- **Interview:** “What does IDF learn, and where?” It weights tokens by training-corpus document frequency; fitting it on test data exposes test distribution information.

## D. Baseline training (`models/train.py`)

- **What:** fit majority, TF-IDF Logistic Regression and TF-IDF Linear SVM candidates using the training partition.
- **Why:** an interpretable cheap baseline distinguishes real learning from class-frequency shortcuts.
- **Concept:** TF-IDF is a sparse representation; logistic cross-entropy and SVM margin losses give different objectives on the same features.
- **Decisions:** word 1–2 grams and optional character 3–5 grams; bounded vocabulary per vectorizer, no automatic class weighting; fixed random seed and max iterations.
- **Common mistakes:** comparing only accuracy to no baseline, unconstrained vocabulary memory, suppressing convergence warnings, calling all classifier outputs confidence.
- **Verify:** fixture training, three validation candidate reports, parameters and distributions in metadata; no test-only token enters fitted vocabulary.
- **When it fails:** inspect empty vocabulary/min_df, tokenization, training class distribution, convergence and feature limits. Confirm source labels aren't one class.
- **Interview:** “Why include character features?” They can generalize spelling fragments/variants but increase sparse dimensions and may memorize identifiers.

## E. Comparison and selection

- **What:** compare three candidates on the same validation partition and choose maximum macro F1, with deterministic simple-model tie-breaking.
- **Why:** selection must be measurable rather than assuming complexity wins.
- **Concept:** validation is used for decisions; test is reserved for estimating performance after those decisions. Repeated validation experiments can themselves overfit.
- **Decisions:** no Transformer without baseline evidence; no hyperparameter sweep on a tiny fixture; no automatic release solely from this selector. Inspect per-class metrics/latency and business constraints manually before release.
- **Common mistakes:** selecting on test accuracy, moving goalposts, comparing candidates on different partitions, ignoring expensive latency/memory for tiny score gains.
- **Verify:** each candidate has validation metrics, only the winner has test metrics; exact configs/fingerprint are recorded.
- **When it fails:** check label order, split identity, class imbalance and baseline parity. Measure resource trade-offs on representative inputs, not demo timing.
- **Interview:** “Why macro F1?” It gives equal class importance, unlike accuracy/weighted averages dominated by large classes; it still may not match asymmetric business costs.

## F. Evaluation (`models/evaluate.py`)

- **What:** compute accuracy, macro precision/recall/F1, weighted F1, per-class report, confusion matrix and one batch latency measurement.
- **Why:** aggregate success can hide severe rare-class failures.
- **Concept:** precision measures false-alarm burden; recall measures misses; F1 is their harmonic mean. Confusion matrix axes use a fixed label order.
- **Decisions:** zero division returns zero, never hidden NaNs; no calibration claim; environment and latency method recorded. Final test is scored only after winner selection, without refitting on validation.
- **Common mistakes:** averaging per-example F1, reading matrix axes backwards, interpreting a single warm batch as p99 API latency, claiming synthetic scores generalize.
- **Verify:** inspect support totals and confusion matrix against split sizes; fixture has six test rows per class. Machine-readable sample is under `reports/`.
- **When it fails:** inspect absent classes, inconsistent labels and sample counts; diagnose infrastructure vs feature transform time separately.
- **Interview:** “Can high test F1 justify automation?” Not without representative data, calibration, asymmetric risk evaluation, distribution-change controls and accountable human oversight.

## G. Error analysis

- **What:** report validation confusion pairs, per-class error counts, short-text errors and example stubs containing label/length but no message content.
- **Why:** experiments need hypotheses about failures, not blind parameter changes.
- **Concept:** error slicing finds systematic model/taxonomy/data problems; correlation with short text is not causal proof.
- **Decisions:** privacy-first text omission rather than unreliable automatic redaction; prioritized follow-up experiments included. No test examples drive iteration.
- **Common mistakes:** publishing real messages, “fixing” test errors then reusing test, asserting uncertain labels are wrong, inferring meaningful patterns from zero synthetic errors.
- **Verify:** JSON count agrees with validation misclassifications; sample fixture has zero errors, which is deliberately uninformative about business failures.
- **When it fails:** inspect safe confusion counts; conduct approved private annotation reviews with independent reviewers and a consistent rubric.
- **Interview:** “How do you distinguish model errors from label errors?” Audit ambiguous examples with domain reviewers, measure agreement, and keep evaluation data corrections versioned.

## H. Priority classification

- **What:** optional separate target, pipeline, candidate selection, metrics and review result.
- **Why:** topic and urgency are not the same label; keyword-created urgency cannot masquerade as ground truth.
- **Concept:** Low/Medium/High may be ordinal; ordinary multiclass loss treats class swaps symmetrically, while ordinal/cost-sensitive methods encode severity distances.
- **Decisions:** disabled by default; explicit verified-label acknowledgment and mapped column required. Start with multiclass baseline; ordinal comparison needs genuine data and a justified cost matrix.
- **Common mistakes:** deriving labels from “urgent”, using resolution time as if available at ticket creation, assuming source priority reflects business need rather than annotator artifacts.
- **Verify:** synthetic priority wiring test checks artifact and API-compatible result. This is not evidence labels exist or a trained priority model is useful.
- **When it fails:** inspect semantics, missing labels, independent groups and split coverage; do not silently fill priority from category.
- **Interview:** “How would you evaluate missing High-priority tickets?” Per-class recall, confusion matrix and business-weighted cost/review analyses, not only global accuracy.

## I. Artifacts (`models/artifacts.py`)

- **What:** persist complete pipelines plus JSON version, provenance fingerprint, mappings/config, parameters, timestamp, environment and validation/test metrics.
- **Why:** saving just coefficients loses the feature vocabulary and preprocessing needed to reproduce inference.
- **Concept:** artifact + code + environment + data/config provenance make a model release traceable. Seeds alone do not guarantee bitwise identity across platforms.
- **Decisions:** local trusted joblib, matching version IDs, unique run output directories, raw datasets and binaries ignored. No signed manifest or remote artifact store is implemented.
- **Common mistakes:** accepting arbitrary pickle uploads, sharing sensitive vocabulary, overwriting prior runs, loading incompatible sklearn versions, treating checksum as authenticity.
- **Verify:** reload prediction tests, mismatched versions reject; metadata stores exact package versions and fingerprint.
- **When it fails:** check paired metadata/artifact, permissions, version compatibility and corruption in a controlled environment. Never retry untrusted deserialization for debugging.
- **Interview:** “Why is joblib unsafe from strangers?” Its pickle machinery can execute code while reconstructing objects, before you can inspect the returned model.

## J. Inference (`models/predict.py`)

- **What:** bounded batch validation, saved-pipeline prediction, task-specific score availability and review flags.
- **Why:** a reusable inference layer prevents web/training code coupling and enables non-HTTP clients.
- **Concept:** closed-set classifiers always choose a known label even for unknown topics; probability is not automatically calibrated certainty or OOD detection.
- **Decisions:** always-review default, mandatory review for synthetic and scoreless SVM results, uncalibrated score naming, fixed demonstration threshold. No irreversible customer actions.
- **Common mistakes:** score=1 from majority being called certainty, adding an untrained “Other”, trusting high scores on unrelated language, changing preprocessing at inference.
- **Verify:** single/batch parity, empty/long inputs, threshold extremes, missing priority and score-interface tests.
- **When it fails:** check artifact config vs API length limits and pipeline type; diagnose vocabulary/domain shift using approved examples, not raw production logs.
- **Interview:** “How would you choose an abstention threshold?” Use separate validation labels, reliability/coverage-risk curves and human-review costs, then freeze policy before final test.

## K. API (`api/`)

- **What:** strict Pydantic single/batch requests, bounded streamed bodies, health/readiness, safe metadata and generated correlation IDs.
- **Why:** model correctness does not guarantee service correctness under malformed or abusive requests.
- **Concept:** liveness differs from readiness; CPU-bound inference is sent to a thread pool rather than blocking the asynchronous request loop.
- **Decisions:** 422 input errors, 413 body cap, 503 unavailable model; redact validation inputs; preserve batch order/optional IDs; no body logging. Concurrency still requires deployment controls.
- **Common mistakes:** assuming Content-Length is truthful, returning stack traces, loading model every request, logging malformed input, wildcard public CORS as “security”.
- **Verify:** TestClient and real HTTP health/ready/predict/batch probes; negative schema/status tests; `/docs` describes response schemas.
- **When it fails:** inspect status and generated request ID, startup events, artifact permissions and model bounds without exposing input text. Requests must be routed to the remote preview, not the browser's localhost.
- **Interview:** “Why can health be 200 while ready is 503?” Process is responsive but the critical model dependency failed to initialize.

## L. Persistence (intentionally omitted)

- **What:** predictions remain transient; optional approved feedback is an external operational design.
- **Why:** avoiding unnecessary text retention reduces privacy/operational risk and prevents predictions being mistaken for labels.
- **Concept:** data minimization and separation of inference, prediction history and independently validated ground truth.
- **Decisions:** no SQLite/Postgres until retention and feedback requirements exist; feedback runbook uses IDs/version/approved labels, not automatic text capture.
- **Common mistakes:** storing full customer messages by default, training on predictions, mixing user metadata into labels without review.
- **Verify:** code writes only training reports/artifacts; inference has no database writes.
- **When it fails:** audit logging and infrastructure first; retention may exist in proxies even if the application has none.
- **Interview:** “When would you add SQLite?” For a local approved minimal prediction/feedback workflow with schema, access control and deletion policy, not just to add a technology badge.

## M. Testing (`tests/`)

- **What:** offline tests cover contracts, preprocessing, groups, training, artifacts, inference, optional priority wiring, API schemas/status/body limits and unavailable models.
- **Why:** small local fixtures catch integration regressions cheaply without private data/network downloads.
- **Concept:** unit tests check properties; integration tests check boundaries; neither estimates generalization.
- **Decisions:** session-scoped trained artifact for speed; temporary directories; synthetic fixture labeled everywhere. Dependency deprecation warning is reported, not hidden.
- **Common mistakes:** asserting fake business performance, downloading large models in unit tests, testing only happy paths, allowing test state to mutate shared config/artifacts.
- **Verify:** `pytest -q`; run all tests after fixes. Actual outcome is in verification record.
- **When it fails:** reproduce individual tests, inspect fixture scope and traceback locally; fix root cause then rerun suite. Initial training metadata and splitting guard bugs were caught and fixed this way.
- **Interview:** “What is not covered?” Real data/license validity, semantic label quality, container execution here, load/security tests and production drift.

## N. Quality and CI

- **What:** package layout, strict YAML config, Ruff lint/format, pinned dependencies, GitHub Actions and wheel packaging.
- **Why:** reproducibility requires a maintainable installable service, not just a notebook.
- **Concept:** CI detects reproducible failures in a clean environment; a checked-in workflow is not proof of a successful CI run.
- **Decisions:** Python 3.11 CI, exact environment pins including build tools; lightweight default offline tests and synthetic CLI integration; no secrets required.
- **Common mistakes:** unpinned transitive dependencies, hidden absolute paths, unexecuted checks claimed passed, leaking credentials in CI output.
- **Verify:** lint/format/test/pip-check locally, build wheel, then inspect actual GitHub run when pushed by authorized workflow.
- **When it fails:** check Python version, lock installation, build backend/wheel availability, dependency conflicts and file packaging.
- **Interview:** “What more is needed than a pip freeze?” Platform constraints, integrity hashes, artifact/image digests, vulnerability review and clean-environment rebuild validation.

## O. Containerization

- **What:** slim Python image runs a non-root API with separately mounted read-only model and readiness healthcheck.
- **Why:** isolate executable environment without baking source data or secrets into image layers.
- **Concept:** image reproducibility and runtime readiness are separate; tags can move, and a successful build doesn't prove serving works.
- **Decisions:** allowlisted build context, UID 10001, 0.0.0.0 binding, no bundled raw data, explicit model path. CI includes build and live probes; local Docker is unavailable.
- **Common mistakes:** root processes, secret-bearing COPY ., treating healthy process as model-ready, mounting unreadable files, claiming an unrun container passed.
- **Verify:** documented `docker build/run`, readiness and prediction HTTP probes on a Docker-capable host.
- **When it fails:** inspect image/package build logs, UID permissions, mount path, artifact compatibility and container health logs. Use only trusted artifact files.
- **Interview:** “Why not train at container startup?” Training complicates startup latency, reproducibility, data access and release control; ship a separately approved artifact.

## P. Monitoring and updates

- **What:** content-free request/prediction events and an operational design for aggregate rates, latency, class/review distribution and delayed ground-truth evaluation.
- **Why:** input distributions and business taxonomies change after deployment.
- **Concept:** covariate/prediction drift is not the same as performance drift; accuracy requires trustworthy observed outcomes.
- **Decisions:** no fake live accuracy; approved external logging/feedback, versioned release gates and rollback. No monitoring platform is unnecessarily added.
- **Common mistakes:** retraining on unreviewed model outputs, using review rate as accuracy, storing secrets in logs, alerting on every short-term fluctuation.
- **Verify:** inspect JSON stdout, model version, synthetic review count; follow `OPERATIONS.md`. Actual live drift/performance collection is not implemented.
- **When it fails:** check event aggregation, volume denominators (requests vs tickets), label delays, privacy access and changing class definitions.
- **Interview:** “What if drift rises but labeled performance stays stable?” Investigate representativeness and slices; don't retrain automatically from distribution distance alone.

## Q. Documentation and presentation

- **What:** reproducible commands, architecture/pipeline diagrams, executed sample reports, explicit limitations and roadmap.
- **Why:** another engineer must be able to audit decisions and reproduce claims.
- **Concept:** model cards/provenance and evidence-backed engineering distinguish demo plumbing from deployed capability.
- **Decisions:** synthetic samples are prominently labeled; no invented users/impact, screenshots or unverified dataset license. Preserve original repository license.
- **Common mistakes:** marketing a perfect toy score as production accuracy, stale commands, omitting failure prerequisites and insecure deployment defaults.
- **Verify:** execute documented CLI commands, compare generated metrics, exercise HTTP; read verification record before repeating claims.
- **When it fails:** fix documentation or implementation together; keep measured and proposed features clearly separated.
- **Interview:** “What is the next highest-value improvement?” A licensed, representative, privacy-approved dataset and clear business taxonomy—not immediately a larger neural model.
