# EV Charge-Session MLOps Pipeline

Drift-aware MLOps pipeline for electric-vehicle charge-session energy and dwell-time prediction.

![status](https://img.shields.io/badge/version-V1%20baseline-blue)
![python](https://img.shields.io/badge/python-3.11-blue)
[![CI](https://github.com/deva-prakash-j/ev-charge-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/deva-prakash-j/ev-charge-mlops/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-105%20collected-brightgreen)
![licence](https://img.shields.io/badge/code-MIT-green)

**Programme:** upGrad MLOps, August 2025 batch · **Author:** Deva Prakash J

---

## Business problem

Workplace and depot EV charging sites predict, at the moment of plug-in, how much energy
a session will draw (kWh) and how long the vehicle stays connected. Those predictions
drive three operational decisions: the "ready-by" estimate shown to the driver, the
site-level load forecast used to stay under a contracted demand limit, and the
smart-charging schedule that allocates limited power across occupied bays.

The model is trained once and then left alone, but its inputs are not stationary. The
connected fleet mix shifts, site occupancy changes, and driver-declared energy requests
drift away from energy actually delivered. The resulting degradation is **silent**: the
API stays up, latency is flat, every dashboard is green, and no alert fires — while
energy forecasts drift and schedules progressively mis-allocate power.

This project replaces uptime monitoring with measured distributional evidence, and
treats the drift detector itself as something to be validated rather than assumed.

## Approach

The dataset contains a precisely dated regime change: workplace charging collapsed in
March–April 2020. That gives the drift detector **ground truth** to be measured against.

| | Feb 2020 | Mar 2020 | Apr 2020 | Change |
|---|---:|---:|---:|---:|
| Site 0001 | 1,402 | 835 | 201 | −86% |
| Site 0002 | 847 | 381 | 24 | −97% |

Volume collapsed *and* the conditional distribution shifted — mean kWh per session rose
(site 0002: 7.33 → 11.79). Both covariate and concept drift, measurable and dated.

## Dataset

**[ACN-Data](https://ev.caltech.edu/dataset)** — real charging sessions from Adaptive
Charging Networks operated with PowerFlex, across three sites (Caltech campus, NASA JPL,
and an office site).

| | |
|---|---|
| Sessions | 66,745 |
| Coverage | Apr 2018 – Sep 2021 |
| Terms | Educational and research use only, agreed when registering for a free API token; citation requested — see [`DATA_LICENCE.md`](DATA_LICENCE.md) |
| PII | None. `userID` is a pseudonymous publisher-assigned identifier — see [`artifacts/pii_audit.json`](artifacts/pii_audit.json) |

**Required citation:**
> Zachary J. Lee, Tongxin Li and Steven H. Low. "ACN-Data: Analysis and Applications of
> an Open EV Charging Dataset." *Proceedings of the Tenth ACM International Conference on
> Future Energy Systems (e-Energy '19)*, 2019.

### Where the real data is

ACN-Data's terms limit use to educational and research purposes and do not address
redistribution, so there is no explicit permission to republish the records. Publishing
them would also let anyone bypass the registration the publisher uses to track usage, and
the records carry a pseudonymous `userID` which, combined with station IDs and
timestamps, would expose per-individual commuting patterns at named workplaces.

The data is therefore shared differently in the two copies of this project:

| | Public GitHub repository | upGrad submission (ZIP / upload) |
|---|---|---|
| Real ACN-Data records | Not included | Included unmodified in `data/raw/`, for evaluation only |
| Extract and reference manifests (counts, SHA-256) | `data/raw/manifest.json`, `data/reference/manifest.json` | Same files |
| Frozen drift reference | Not included | Generated in `data/reference/` from the licensed extract |
| Synthetic fixture | `data/sample/` | Same files |

See [`DATA_LICENCE.md`](DATA_LICENCE.md) for the publisher's terms and the conditions
attached to the evaluation copy.

The public repository ships instead:

- the **retrieval script**, so acquisition is reproducible;
- a **manifest** with SHA-256 hashes and exact row counts, so provenance is verifiable;
- a **synthetic fixture** (`data/sample/`) generated from aggregate statistics only.

The fixture reproduces the schema and the marginal distributions closely enough that the
pipeline is fully runnable with **no credentials and no data access**:

| | Real (21,364) | Synthetic (2,579) |
|---|---:|---:|
| kWh mean | 12.74 | 13.42 |
| kWh median | 9.92 | 9.25 |
| userInputs present | 88.4% | 88.3% |
| weekday share | 95.5% | 95.1% |
| arrive 05:00–10:00 | 65.2% | 64.8% |

Known fixture limitations: median connected hours is 5.08 vs 6.89 (the lognormal underfits
that tail), and the delivered-vs-requested relationship is log-linear by construction, so
tree ensembles have less to exploit on the fixture than on real data. Headline results in
the technical report use the licensed extract.

## Baseline results

Trained on the licensed extract. Temporal split: train ≤ 2019-12-31, validation
2020-01-01..2020-02-29, and the **same fitted model** scored on 2020-04-01..2020-06-30 to
separate drift impact from model quality.

The licensed records are not in the public repository, so this table cannot be retrained
from a public clone alone; the upGrad submission includes them in `data/raw/`. Its
data-free provenance record, including the training-frame fingerprint, row counts and
exact scores, is tracked in
[`benchmarks/licensed_baseline.json`](benchmarks/licensed_baseline.json). To reproduce it,
follow [Licensed-data path](#licensed-data-path--upgrad-submission-copy) and compare the
resulting `data_sha256` and split counts before comparing metrics. A different
fingerprint means the result is not an exact reproduction. The credential-free sample
path is independently runnable, but its metrics are a pipeline demonstration, not a
replacement for the licensed-data result.

| Model | Train MAE | Validation MAE | Post-shift MAE | Validation R² | Post-shift R² |
|---|---:|---:|---:|---:|---:|
| Mean (floor) | 7.067 | 7.894 | 8.479 | −0.010 | −0.169 |
| Ridge | 4.979 | 5.382 | 6.944 | 0.501 | 0.235 |
| **HistGradientBoosting** | **3.701** | **4.399** | **6.272** | **0.579** | **0.355** |

Two things matter here:

1. The candidate beats the mean floor by **44%** on validation, so the added complexity is
   earning its place rather than being assumed.
2. Scored on post-shift data the **same model** degrades from 4.399 to 6.272 MAE — a **43%
   increase in error** — and R² falls from 0.579 to 0.355. No uptime metric, latency graph
   or error-rate dashboard would have shown any of this.

That second row is the whole argument for V2: the failure is real, it is large, and it is
invisible to conventional monitoring.

Top features by permutation importance on validation: `kWhRequested` (2.54),
`milesRequested` (1.58), `declared_duration_minutes` (0.48), `siteID` (0.22).

### Connected-duration model

The same 66,743 post-quarantine sessions and temporal boundaries are used to predict
`connectedHours` (hours from connection to disconnect). MAE is in hours:

| Model | Validation MAE | Post-shift MAE |
|---|---:|---:|
| Mean | 3.074 | 2.835 |
| Ridge | 1.982 | 2.393 |
| **HistGradientBoosting** | **1.659** | **2.324** |

This is a separate target, not a feature: the dwell model uses only the same plug-in-time
fields as the energy model. The synthetic demo trains the same target but its scores are
not evidence of licensed-data model quality. Full-data evaluation is recorded in
[`benchmarks/dwell_baseline.json`](benchmarks/dwell_baseline.json); the full run report is
`artifacts/training_dwell_raw.json` in the upGrad copy.

## Installation

Requires Python 3.11.

**Windows (PowerShell)**

```powershell
git clone https://github.com/deva-prakash-j/ev-charge-mlops.git
cd ev-charge-mlops
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

**Linux / macOS**

```bash
git clone https://github.com/deva-prakash-j/ev-charge-mlops.git
cd ev-charge-mlops
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

Substitute `.venv/bin/python` for `.\.venv\Scripts\python.exe` in every command below.

## Execution

### Demo path — no credentials required

One command runs the whole pipeline against the committed synthetic fixture:

```powershell
.\.venv\Scripts\python.exe -m evcharge.pipeline --source sample
```

```
normalise  ok    2,579 rows, 3 sites, 2019-07..2020-06
contract   ok    2,579 clean, 0 quarantined, 16/16 checks passed
train      ok    energy best: hist_gradient_boosting; dwell best: hist_gradient_boosting
                 (synthetic validation MAE: 6.342 kWh; 3.700 hours)
```

No API token, no manual edits. Stages can also be run individually:

```powershell
.\.venv\Scripts\python.exe -m evcharge.ingest.normalise    --source sample
.\.venv\Scripts\python.exe -m evcharge.validation.contract --source sample
.\.venv\Scripts\python.exe -m evcharge.models.train        --source sample
.\.venv\Scripts\python.exe -m pytest -q
```

Exploratory analysis, also credential-free:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks/01_data_understanding.ipynb
```

Experiment runs are written to `mlruns/`; browse them with `mlflow ui`.

### Prefect workflow automation

Install the optional orchestration and monitoring dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[orchestration,monitoring]"
```

Run the contract-gated training flow once, or validate newly added monthly batches
without fetching:

```powershell
.\.venv\Scripts\python.exe -m evcharge.orchestration train --source sample
.\.venv\Scripts\python.exe -m evcharge.orchestration monitor --source raw --no-fetch
```

To run the scheduled raw-data monitor locally, start a Prefect server in one terminal:

```powershell
.\.venv\Scripts\prefect.exe server start
```

In another terminal, configure its local API and start the deployment process:

```powershell
.\.venv\Scripts\prefect.exe config set PREFECT_API_URL=http://127.0.0.1:4200/api
.\.venv\Scripts\python.exe -m evcharge.orchestration serve
```

The deployment fetches the licensed ACN extract and runs at 02:00 UTC on the first of
each month. It normalises the data and validates each new or changed month; content
fingerprints in `artifacts/monitoring/monthly_raw.json` let unchanged batches skip
validation and scoring, while failed scoring or retraining is retried. Fetching requires
the ACN token in `.env`. This local schedule runs only while both Prefect server and
deployment process are up.

The flow scores each newly validated raw month against the checksum-verified frozen
reference, exports per-feature PSI and Jensen–Shannon, numeric KS, sample counts,
missingness and alert state through Pushgateway, then invokes retraining only for an alert
on the latest available month. The retraining hook uses the existing contract and MLflow
promotion gates; Prometheus/Grafana alerts are for operator visibility and do not trigger
a webhook.

### Drift monitoring and evaluation

The Compose stack includes Pushgateway, Prometheus and Grafana. Start all services (use
an alternate API host port if 8000 is occupied):

```powershell
$env:EVCHARGE_API_PORT = '8001'
docker compose up --build -d --wait
docker compose ps
```

Grafana is at `http://127.0.0.1:3000` and provisions the `EV Charge Drift Monitoring`
dashboard with the Prometheus datasource. The local dashboard password is controlled by
`GRAFANA_ADMIN_PASSWORD` (defaults to `admin` for this loopback-only academic demo).
Prometheus and Pushgateway are at `http://127.0.0.1:9090` and
`http://127.0.0.1:9091`. The Prefect process runs on the host and uses
`PUSHGATEWAY_URL`, defaulting to `http://127.0.0.1:9091`.

The licensed-data evaluation and its limitations are recorded in
[`docs/drift_evaluation.md`](docs/drift_evaluation.md). Reproduce the declared detector
study and the separate held-out recovery experiment with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_drift.py
.\.venv\Scripts\python.exe scripts\evaluate_retraining_recovery.py
```

Both commands require the licensed normalized raw partitions, frozen reference, and a
raw Production energy model. The detector alerts on April 2020 delivered-energy PSI,
with zero leave-one-month-out false alarms and a first controlled sensitivity crossing
at +30% energy shift. The 10% post-shift recovery target was **not** met by either
chronological retraining experiment; the measured failure is documented rather than
presented as a successful recovery.

### Model registry and promotion gate

Each training run registers its best-validation model under a source/target-specific
name (`evcharge-sample-energy`, `evcharge-raw-energy`, or the corresponding `-dwell`
name). The new version is assigned the `staging` alias. It reaches the `production`
alias only when its held-out validation MAE beats both the mean and Ridge baselines
and is no more than 10% worse than the current Production model scored on those same
validation rows. The first model can bootstrap Production only by clearing both
baseline floors. Blocked candidates remain in Staging, and every decision is recorded
in `artifacts/promotion_decisions.jsonl` and the training report.

To verify the safety gate after training, submit a deliberately poor constant-zero
candidate. The command should report that it was blocked; it never changes Production:

```powershell
.\.venv\Scripts\python.exe -m evcharge.models.registry --source sample --target energy --prove-degraded-candidate
```

### Continuous integration

GitHub Actions runs Ruff, the test suite, the sample-data contract and temporal
evaluation on every push and pull request, then builds the Docker image. The sample
pipeline promotes eligible energy and dwell models in a run-local MLflow registry.
CI next registers a deliberately degraded candidate and fails if it is not blocked
or if the Production alias changes. The workflow uploads the MLflow runs, promotion
decisions and reports as a 14-day artifact. This is a safety check, not a deployment:
CI does not publish a model or require registry credentials.

### Model artefacts

Training writes the selected energy model to `models/<source>/` and the
connected-duration model to `models/<source>_dwell/`:

| File | Purpose |
|---|---|
| `model.joblib` | Fitted estimator |
| `feature_spec.json` | Category vocabulary and column order it was fitted with |
| `metrics.json` | Full slice-wise evaluation |
| `MODEL_CARD.md` | Target, provenance, git commit, headline metrics |

The model and its feature spec are saved together deliberately — the estimator alone
cannot reproduce its own input encoding. `models/sample/` is committed so an evaluator has
a concrete artefact without running training first; `models/raw/` is gitignored.

The trainer also supports connected-duration prediction:

```powershell
.\.venv\Scripts\python.exe -m evcharge.models.train --source raw --target dwell
```

This predicts `connectedHours` from the same plug-in-time feature set and writes
`models/raw_dwell/` plus `artifacts/training_dwell_raw.json`. Duration and other
post-session measurements never enter the feature matrix. See
[`docs/feature_specification.md`](docs/feature_specification.md) for the complete feature
rationale and the documented occupancy deferral.

```python
import joblib
from evcharge.features.build import FeatureSpec

model = joblib.load("models/sample/model.joblib")
spec = FeatureSpec.load("models/sample/feature_spec.json")
```

### Local prediction API

The saved sample model is served by FastAPI using the same feature builder as training:

```powershell
.\.venv\Scripts\python.exe -m uvicorn evcharge.serve.app:app --host 127.0.0.1 --port 8000
```

`GET /health` reports readiness. `POST /predict` accepts site and station identifiers,
`connectionTimeLocal`, optional UTC `connectionTime` / `requestedDeparture`, and the
declared request fields (`kWhRequested`, `milesRequested`, `minutesAvailable`,
`WhPerMile`). Unknown category values are handled by the fitted feature spec; post-hoc
fields are rejected. The demo service binds to localhost and does not implement
authentication, TLS termination, or production deployment hardening.

### Docker Compose registry-backed API

Docker Compose starts a local MLflow server and the API. On an empty registry volume, a
one-shot bootstrap container runs the sample pipeline and registers a model before the API
starts. Later starts reuse the persisted Production alias. The container image includes
the synthetic sample only; it excludes licensed `data/raw/` records. First-time build and
training may take several minutes.

```powershell
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

The API loads `evcharge-sample-energy@production` from MLflow rather than
`models/sample/`. The MLflow UI is available at `http://127.0.0.1:5000`. Both host ports
bind to loopback only; the local demo does not provide authentication or TLS. If port 8000
or 5000 is already in use, choose other host ports before starting Compose, for example
`$env:EVCHARGE_API_PORT = '8001'` or `$env:EVCHARGE_MLFLOW_PORT = '5001'`, and use those
ports in the commands below.

Use the checked-in sample request:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType 'application/json' -InFile .\examples\predict_request.json
Invoke-WebRequest http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content
```

Example response from a fresh Compose registry (`predicted_kWhDelivered` varies with
the promoted model version):

```json
{
  "predicted_kWhDelivered": 14.23,
  "model_name": "evcharge-sample-energy",
  "model_version": "1"
}
```

The API exports Prometheus counters and histograms at `/metrics`. Each successful
prediction is also emitted as a structured JSON log and appended to the persistent
`prediction_logs` volume at `/var/log/evcharge/predictions.jsonl`. The record contains
the encoded feature vector and prediction, not the raw site or station identifiers.
Inspect container logs with `docker compose logs -f api`.
The default application image is `evcharge-capstone:local`; set `EVCHARGE_IMAGE` to an
existing tag to deploy or restore the shared image used by the API, MLflow, and bootstrap
containers. The isolated rollback drill and exact procedure are recorded in
[`docs/operations_governance.md`](docs/operations_governance.md).

Run the container smoke test after Compose reports the API healthy:

```powershell
$env:EVCHARGE_API_URL = 'http://127.0.0.1:8000'
.\.venv\Scripts\python.exe -m pytest -q tests\test_container_smoke.py
Remove-Item Env:EVCHARGE_API_URL
```

`docker compose down` stops services but preserves the MLflow and prediction-log volumes.
Use `docker compose down -v` only when you intentionally want to erase those local
registry and log records.

### Licensed-data path — upGrad submission copy

The upGrad submission already contains the extract in `data/raw/`, so no token is needed.
Check the files against the manifest, then run the pipeline without `--fetch`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_fetch_sessions.py -q -rs   # SHA-256, bytes, sessions vs data/raw/manifest.json
.\.venv\Scripts\python.exe -m evcharge.pipeline --source raw
.\.venv\Scripts\python.exe -m evcharge.validation.reference --source raw
```

`artifacts/training_raw.json` then reports `data_sha256` and `rows`; they should equal
`training_frame_sha256` and `split_rows` in
[`benchmarks/licensed_baseline.json`](benchmarks/licensed_baseline.json), and the
headline metrics follow from them. Compare `artifacts/training_dwell_raw.json` against
[`benchmarks/dwell_baseline.json`](benchmarks/dwell_baseline.json) for the dwell target.
The reference command creates the frozen 2019-07..2020-02 drift snapshot and verifies its
metadata in `data/reference/manifest.json`. The snapshot file itself is included only in
the upGrad evaluation copy.

### Full path — requires an ACN-Data token

```powershell
Copy-Item .env.example .env      # then paste your token into ACN_API_TOKEN
.\.venv\Scripts\python.exe -m evcharge.ingest.check_access
.\.venv\Scripts\python.exe -m evcharge.pipeline --source raw --fetch
.\.venv\Scripts\python.exe -m evcharge.validation.pii_audit
```

`.env` is gitignored. The token is never logged — it is wrapped in a `Secret` type whose
`repr` and `str` are masked, with a test asserting it cannot leak.

### Regenerating the fixture

```powershell
.\.venv\Scripts\python.exe -m evcharge.fixtures.profile    # needs the licensed extract
.\.venv\Scripts\python.exe -m evcharge.fixtures.generate   # needs only the profile YAML
```

## Repository structure

```
ev-charge-mlops/
├── configs/                      Declarative configuration, no secrets
│   ├── data.yaml                 Schema contract, leakage controls, split boundary
│   ├── contract.yaml             Expectations and quarantine rules
│   ├── features.yaml             Feature specification
│   ├── model.yaml                Models, splits, tracking
│   └── sample_profile.yaml       Aggregate statistics for fixture generation
├── data/
│   ├── sample/                   Synthetic fixture (committed)
│   ├── raw/ interim/ quarantine/ Licensed extract and derivatives (gitignored except
│   │                             raw/manifest.json; raw/ is in the upGrad ZIP)
│   └── reference/                Frozen drift window; manifest tracked, data in ZIP only
├── notebooks/
│   └── 01_data_understanding.ipynb   EDA; imports src/, never redefines logic
├── models/
│   ├── sample/                   Trained model, feature spec, metrics, model card
│   ├── raw/                      Energy model trained on licensed data (gitignored)
│   └── raw_dwell/                Dwell model trained on licensed data (gitignored)
├── src/evcharge/
│   ├── config.py                 Settings, secret handling, YAML loading
│   ├── pipeline.py               Single entry point: normalise → contract → train
│   ├── ingest/                   API client, retrieval, normalisation
│   ├── validation/               PII audit, data contract
│   ├── features/                 Shared builder used by training and serving
│   ├── models/                   Baselines, temporal evaluation, MLflow
│   └── fixtures/                 Aggregate profiling, synthetic generation
├── docs/architecture.md          V1 and implemented V2 diagrams
├── monitoring/                   Prometheus rules and Grafana provisioning
├── scripts/                      CI gate and drift/recovery evaluations
├── STACK.md                      Tool choices against named alternatives
├── tests/                        105 tests (container smoke test is opt-in)
├── mlruns/                       MLflow tracking store (gitignored)
└── artifacts/                    Run reports (pii_audit.json committed as evidence)
```

## Data quality findings

The contract was built against defects actually present in the feed, not invented ones.

| Finding | Scale | Handling |
|---|---|---|
| `siteID` emitted as both int `2` and str `"0002"` | 2,081 rows | Coerced; blocking expectation |
| `clusterID` mixed int/str | crashed Parquet write | Coerced; blocking expectation |
| Sessions of 245h and 214h | 2 rows | Quarantined with reason, not dropped |
| Done-charging timestamp order, station namespace and declared/delivered plausibility | 16 data-contract checks total | Site-specific identifier domains; timestamp order checked on complete pairs with documented historical tolerances; gross energy ratios over 3:1 must be below 0.5% |

Quarantining rather than dropping keeps upstream defects visible — a 10-day session
delivering 1.48 kWh is a real signal, not noise to be silently discarded.

### Leakage controls

Two traps are enforced in code and pinned by tests:

- **`userInputs` is a list of driver revisions.** Only the earliest entry exists at
  plug-in time; later revisions leak the future. `min(modifiedAt)` is taken.
- **`disconnectTime` and `doneChargingTime` are post-hoc.** They are listed as
  `post_hoc_columns` in `configs/data.yaml` and may never enter the feature matrix for
  a plug-in-time model.

Train/test splitting is **temporal, never random** — a random split leaks across the
2020 regime change and would produce a flattering, meaningless score.

## Stack decisions

**HistGradientBoosting instead of LightGBM.** LightGBM was the original choice, but on
Windows it crashes the process with an access violation whenever `pyarrow` is imported
first — a conflict between pyarrow's bundled OpenMP runtime and LightGBM's. Since this
pipeline reads Parquet, that import order is unavoidable. Depending on import order would
leave a segfault waiting for anyone who runs this on Windows, so the model was switched to
scikit-learn's `HistGradientBoostingRegressor`: the same algorithm family, native NaN
support, and no additional native runtime to conflict.

**No imputation for the gradient-boosted model.** Roughly 27% of sessions have no driver
request at all. `has_user_inputs` makes that explicit and the estimator handles NaN
natively, so missingness is treated as signal rather than silently filled with a median.

## Operational ownership and rollback

The proposal identifies these model consumers: charging operations/site energy
management, the charging app/driver-experience team, the smart-charging scheduler, and
data governance/audit. The MLOps engineer owns deployment, promotion, rollback,
monitoring, and retraining; charging operations accepts or rejects retrained candidates.

For this academic demo, **Deva Prakash J** is the acting MLOps owner and first alert
responder. A real production deployment still requires an explicitly staffed on-call rota
and a named charging-operations approver; this prototype is not a production on-call
service. The isolated rollback drill passed and restored the sample API to registry
version 1 using the captured previous image ID. See
[`docs/operations_governance.md`](docs/operations_governance.md) for scope, evidence, and
limitations.

## Current status — V1 baseline and V2 monitoring

| Component | Status |
|---|---|
| Data acquisition from ACN-Data API | Done |
| PII audit with structure-aware detection | Done |
| Normalisation, local-time monthly partitioning | Done |
| Data contract (16 checks) + quarantine | Done |
| Synthetic fixture for credential-free execution | Done |
| Configuration layer | Done |
| Shared feature module with the 1,000-session API parity audit and leakage tests | Done |
| Baseline models and temporal evaluation | Done |
| MLflow experiment tracking | Done |
| MLflow Registry aliases and validation promotion gates | Done |
| Single-command pipeline entry point | Done |
| Local FastAPI prediction and health endpoints | Done |
| Docker Compose Production-alias API, `/metrics`, and JSONL prediction logs | Done; verified end to end with Docker Compose |
| EDA notebook | Done |
| Model artefacts and model card | Done |
| Connected-duration (dwell-time) prediction | Done; separate training target and evaluation |
| Architecture diagrams and `STACK.md` | Done |
| Test suite | Done — 105 passed, including the live container smoke test |
| GitHub Actions CI — Ruff, tests, contract, registry promotion gate, Docker build | Done; runs on every push and pull request |
| Prefect training and monthly monitoring flows, drift scoring and guarded retraining | Done; requires local Prefect server and licensed raw data |
| Evidently PSI/KS/JS, Prometheus export, Grafana dashboard and alert rules | Done; target PSI alert validated on April 2020 |
| Drift false-alarm, sensitivity and post-shift recovery evaluation | Done; 0/8 false alarms, +30% first tested PSI crossing; recovery target not met |
| Phase 7 governance | Done; isolated registry-alias and prior-image rollback drill passed; ownership and direct-tool license inventory documented |

## Remaining work — Submission and model recovery

- **Submission** — technical report, executive presentation, final packaging and
  fresh-clone/fresh-ZIP verification.
- **Model recovery** — investigate why later chronological retraining windows remain
  above the pre-shift MAE target (see [`docs/drift_evaluation.md`](docs/drift_evaluation.md)).

The Phase 3 local deployment is intentionally unauthenticated and binds its host ports to
loopback. It is an evaluation/demo deployment, not a public production endpoint.

## Licence

Code is MIT licensed. The ACN-Data dataset is **not** covered by that licence and remains
subject to the publisher's terms (see [`DATA_LICENCE.md`](DATA_LICENCE.md)); it is not
published in this repository. Third-party package and tool licenses, including the
inventory scope and exclusions, are listed in [`STACK.md`](STACK.md).
