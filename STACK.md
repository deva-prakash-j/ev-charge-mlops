# STACK

Every tool here is justified against a named alternative. Entries marked **(V2)** are
planned rather than implemented.

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

Writes go to a `.staging` tree and are swapped atomically. This was added after a mid-write
failure left 44,327 of 66,745 rows on disk and the next run silently read the partial tree.

---

## Experiment tracking

### MLflow

**Alternatives considered:** Weights & Biases, CSV logs.

W&B is better hosted but requires an account, violating the local-only constraint. MLflow
runs against a local `mlruns/` directory with no server, and its Model Registry is needed
for the V2 promotion and rollback workflow — so adopting it now avoids a migration later.

Each run logs the git commit and a SHA-256 of the training input, so any metric can be
traced back to the exact code and data that produced it.

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

SDV is the better general tool and is planned for V2 to generate drift of **known
magnitude** for detector sensitivity testing. For the committed fixture it was overkill:
the requirement is a small, deterministic, dependency-light artefact whose generation an
evaluator can read and verify in one file.

The split matters legally. `fixtures/profile.py` needs the licensed extract and produces
only aggregate statistics; `fixtures/generate.py` reads those statistics and nothing else.
No real record is ever redistributed.

The generator derives delivered energy from the driver's request using a fitted log-log
relationship. Without that, the fixture reproduced correct marginals but no feature-target
relationship, and a model trained on it could not beat the mean baseline — making the demo
path actively misleading.

---

## Planned (V2)

| Tool | Purpose | Alternative considered |
|---|---|---|
| FastAPI + Uvicorn | Model serving | Flask — no async, weaker schema validation |
| Docker + Compose | Environment parity | Conda — does not capture OS-level dependencies |
| GitHub Actions | CI and quality gates | Jenkins — needs a server to maintain |
| Prefect | Orchestration | Airflow — heavier scheduler for a four-week project |
| Evidently | Drift reports and metrics | Hand-rolled PSI — no Data Docs, more to verify |
| Prometheus + Grafana | Metric storage and alerting | Logging to files — no alerting path |
| SDV | Injected drift for detector testing | — |
