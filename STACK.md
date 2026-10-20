# STACK

Every tool here is justified against a named alternative. Entries marked **(V2)** are
planned rather than implemented; Prefect orchestration and the local FastAPI serving path
are implemented.

## Constraints

Open-source and locally runnable; no cloud account, paid tier or GPU; reproducible by an
independent evaluator on a clean machine with no credentials.

---

## Modelling

### HistGradientBoostingRegressor (scikit-learn)

**Alternative considered:** LightGBM.

LightGBM was the original choice and is the stronger library in isolation. It was removed
after it crashed the process with an access violation:

```
OSError: exception: access violation reading 0x0000000000000000
  lightgbm/basic.py  LGBM_DatasetSetField
```

Bisecting ruled out index contiguity, NaN values, value ranges, dataset size and
scikit-learn import order. The actual trigger is import order with **pyarrow**:

| Import order | Result |
|---|---|
| `pyarrow` then `lightgbm` | access violation |
| `lightgbm` then `pyarrow` | works |

A conflict between pyarrow's bundled OpenMP runtime and LightGBM's. Because the pipeline
reads Parquet, pyarrow is always imported, and relying on import order would leave a
segfault waiting for any evaluator on Windows — failing silently and catastrophically
rather than loudly.

`HistGradientBoostingRegressor` is the same algorithm family (it is itself LightGBM-derived),
handles NaN natively, and ships inside scikit-learn, so it removes the entire class of
failure. The measured cost is negligible at this data scale.

### Mean and Ridge baselines

**Alternative considered:** reporting only the candidate model's metrics.

A single headline number cannot show whether complexity is earning its place. With floors
in place, the 44% validation-MAE improvement over the mean predictor is measured rather
than claimed.

---

## Data validation

### Great Expectations

**Alternatives considered:** Pandera, hand-rolled assertions.

Pandera is lighter and more ergonomic, but Great Expectations produces a serialisable
validation result with per-expectation success, unexpected counts and percentages — which
is what makes a gate auditable rather than just a pass/fail boolean. The declarative suite
also lives in `configs/contract.yaml`, so changing the contract is a config change.

**Cost:** GE 0.18 pulls in deprecated marshmallow and pyparsing calls, which are filtered
in `pyproject.toml`.

### Structured quarantine rules

**Alternative considered:** pandas `query()` expression strings in config.

Expression strings are more flexible but evaluate config content as code. Rules are instead
`{column, op, value}` triples resolved against a fixed operator table, so no configuration
value is ever executed.

---

## Storage

### Parquet, partitioned by site and local month

**Alternatives considered:** CSV, a single Parquet file, SQLite.

Monthly partitions mirror the cadence at which ACN-Data publishes, which makes batch-wise
drift evaluation a directory read rather than a filter. Partitioning uses **site-local**
time so "March 2020" means the operator's March, not GMT's.

Writes go to a `.staging` tree. The prior tree is moved to a uniquely named backup before
installation; a failed install restores it, and a later run recovers the newest backup if
the process stopped during the swap. This was added after a mid-write failure left 44,327
of 66,745 rows on disk and the next run silently read the partial tree.

---

## Experiment tracking

### MLflow

**Alternatives considered:** Weights & Biases, CSV logs.

W&B is better hosted but requires an account, violating the local-only constraint. MLflow
runs against a local `mlruns/` directory with no server. The Model Registry is used for
candidate registration and quality-gated promotion.

Each run logs the git commit and a SHA-256 of the training input, so any metric can be
traced back to the exact code and data that produced it.

The trainer uses the Registry's `staging` and `production` aliases rather than MLflow
stages, which are deprecated in MLflow 2.9 and later. The best-validation candidate is
registered and assigned to `staging`; the `production` alias changes only if the
candidate beats both mean and Ridge baseline MAE and stays within 10% of the current
Production model's MAE on the same validation rows. The first candidate may bootstrap
Production only after clearing both baseline floors. Promotion decisions and their data
and code fingerprints are written to `artifacts/promotion_decisions.jsonl`.

---

## Configuration and secrets

### YAML configs + `.env` via python-dotenv

**Alternatives considered:** Pydantic Settings, Hydra.

Hydra is powerful but its composition model is more machinery than this pipeline needs.
Pydantic Settings would add validation, which `config.py` already performs for the few
fields involved.

The API token is wrapped in a `Secret` type whose `repr` and `str` are masked, with a test
asserting the value cannot appear in either. This matters because tracebacks and log lines
routinely serialise config objects.

---

## Synthetic fixture

### Hand-written generator from an aggregate profile

**Alternative considered:** SDV (Synthetic Data Vault).

SDV was considered for detector sensitivity testing but is not used. The measured
sensitivity curve applies explicit multiplicative shifts to `kWhDelivered`, making the
intervention deterministic and easy to inspect without another synthetic-data
dependency.

The split keeps real records out of the public repository. `fixtures/profile.py` needs
the licensed extract and produces only aggregate statistics; `fixtures/generate.py` reads
those statistics and nothing else, so the committed fixture contains no real record.

The generator derives delivered energy from the driver's request using a fitted log-log
relationship. Without that, the fixture reproduced correct marginals but no feature-target
relationship, and a model trained on it could not beat the mean baseline — making the demo
path actively misleading.

---

## Local model serving

### FastAPI + Uvicorn

The local development API loads the saved model and its fitted `FeatureSpec` together,
then calls the shared feature builder used by training. The Docker Compose API instead
loads the `production` alias from MLflow and waits for the sample registry bootstrap on
first start. Both expose `/health`, `/predict` and Prometheus-format `/metrics`; the
container API emits a feature-only prediction JSONL record for later drift analysis.

The Compose image uses the exact direct runtime pins in `requirements-container.txt`,
excludes licensed records, and persists the MLflow store and prediction logs in named
volumes. Host ports bind to loopback. The demo has no authentication or TLS and is not
intended as a public production deployment.

---

## Drift monitoring

### Evidently

**Alternative considered:** hand-rolled PSI and statistical tests.

Evidently runs PSI and Jensen–Shannon distance for numeric and categorical features
against the checksum-verified frozen reference; the KS test is applied to continuous
features. SciPy supplies the KS statistic alongside Evidently's p-value. The monitor
writes data-free JSON, Evidently JSON and HTML reports; it also records non-missing
sample counts and missing fractions so scores are interpreted with coverage. Evidently
is an optional dependency (`.[monitoring]`).

### Prometheus Pushgateway, Prometheus and Grafana

**Alternative considered:** application logs alone.

The monthly batch monitor is a short-lived Prefect flow, so it pushes its feature scores
to the Pushgateway rather than trying to expose them from the always-on prediction API.
Prometheus scrapes both the Pushgateway and API, evaluates PSI and API-availability rules,
and retains 30 days of local data. Grafana provisions a Prometheus datasource and
dashboard for target/prediction PSI, input PSI, KS, Jensen–Shannon, alert state, sample
counts, missingness and API prediction rate. Compose ports bind to loopback; the stack is
for local academic evaluation, not public production.

The Prefect monitor directly calls the quality-gated retraining flow when the latest
scored batch alerts. Prometheus/Grafana provide operator visibility; no alert webhook or
implicit model promotion is involved.

---

## CI/CD

GitHub Actions runs Ruff, the tests, sample-data validation and evaluation, and a Docker
build on pushes and pull requests. Its local MLflow registry promotes eligible sample
models, then verifies a deliberately degraded candidate is blocked without changing the
Production aliases. The workflow uploads the run store and decision reports for 14 days;
it does not publish models to a persistent registry or deploy the API.

---

## Workflow orchestration

### Prefect 3

**Alternative considered:** Apache Airflow.

Prefect orchestrates the existing contract-gated training pipeline and schedules a
monthly raw-data monitor. The monitor fetches ACN-Data, normalises it, validates and
scores each new or changed month, and exports the drift report. Content fingerprints
prevent repeated validation and scoring; score and retraining state are tracked
separately so failures can be retried without duplicating successful work. A retraining
flow accepts the detector's boolean alert and runs the existing training, evaluation,
and MLflow promotion gates; it does not bypass them.

Prefect is an optional dependency (`.[orchestration]`), so API and batch-only users do
not install its scheduler. The schedule is local: a Prefect server and the flow-serving
process must remain running. This is appropriate for the academic single-machine
deployment; a managed Prefect service is an alternative if always-on scheduling is
required. The monthly deployment runs at 02:00 UTC on day 1 and requires an ACN API token.

Detector validation and the failed 10% post-shift recovery target are documented in
[`docs/drift_evaluation.md`](docs/drift_evaluation.md).
