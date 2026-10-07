# EV Charge-Session MLOps Pipeline

Drift-aware MLOps pipeline for electric-vehicle charge-session energy prediction.

![status](https://img.shields.io/badge/version-V1%20baseline-blue)
![python](https://img.shields.io/badge/python-3.11-blue)
![tests](https://img.shields.io/badge/tests-42%20passing-brightgreen)
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
| Licence | Open research use; free registration for an API token, citation required |
| PII | None. `userID` is a pseudonymous publisher-assigned identifier — see [`artifacts/pii_audit.json`](artifacts/pii_audit.json) |

**Required citation:**
> Zachary J. Lee, Tongxin Li and Steven H. Low. "ACN-Data: Analysis and Applications of
> an Open EV Charging Dataset." *Proceedings of the Tenth ACM International Conference on
> Future Energy Systems (e-Energy '19)*, 2019.

### Why the real data is not in this repository

ACN-Data is distributed under publisher terms that require registration and do not grant
redistribution. The records also contain a pseudonymous `userID` which, combined with
station IDs and timestamps, would expose per-individual commuting patterns at named
workplaces.

So this repository ships instead:

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

Known fixture limitation: median connected hours is 5.08 vs 6.89 — the lognormal
underfits that tail. Headline results in the technical report use the licensed extract.

## Installation

Requires Python 3.11.

```powershell
git clone https://github.com/deva-prakash-j/ev-charge-mlops.git
cd ev-charge-mlops
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

On Linux/macOS substitute `python3.11 -m venv .venv` and `.venv/bin/python`.

## Execution

### Demo path — no credentials required

```powershell
.\.venv\Scripts\python.exe -m evcharge.ingest.normalise   --source sample
.\.venv\Scripts\python.exe -m evcharge.validation.contract --source sample
.\.venv\Scripts\python.exe -m pytest -q
```

Runs against the committed synthetic fixture. No API token, no manual edits.

### Full path — requires an ACN-Data token

```powershell
Copy-Item .env.example .env      # then paste your token into ACN_API_TOKEN
.\.venv\Scripts\python.exe -m evcharge.ingest.check_access
.\.venv\Scripts\python.exe -m evcharge.ingest.fetch_sessions
.\.venv\Scripts\python.exe -m evcharge.validation.pii_audit
.\.venv\Scripts\python.exe -m evcharge.ingest.normalise    --source raw
.\.venv\Scripts\python.exe -m evcharge.validation.contract --source raw
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
│   └── sample_profile.yaml       Aggregate statistics for fixture generation
├── data/
│   ├── sample/                   Synthetic fixture (committed)
│   └── raw/ interim/ quarantine/ Licensed extract and derivatives (gitignored)
├── src/evcharge/
│   ├── config.py                 Settings, secret handling, YAML loading
│   ├── ingest/                   API client, retrieval, normalisation
│   ├── validation/               PII audit, data contract
│   └── fixtures/                 Aggregate profiling, synthetic generation
├── tests/                        42 tests
└── artifacts/                    Run reports (pii_audit.json committed as evidence)
```

## Data quality findings

The contract was built against defects actually present in the feed, not invented ones.

| Finding | Scale | Handling |
|---|---|---|
| `siteID` emitted as both int `2` and str `"0002"` | 2,081 rows | Coerced; blocking expectation |
| `clusterID` mixed int/str | crashed Parquet write | Coerced; blocking expectation |
| Sessions of 245h and 214h | 2 rows | Quarantined with reason, not dropped |

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

## Current status — V1 (Baseline)

| Component | Status |
|---|---|
| Data acquisition from ACN-Data API | Done |
| PII audit with structure-aware detection | Done |
| Normalisation, local-time monthly partitioning | Done |
| Great Expectations data contract + quarantine | Done |
| Synthetic fixture for credential-free execution | Done |
| Configuration layer | Done |
| Test suite | Done — 42 tests |
| Feature engineering module + parity tests | In progress |
| Baseline model training and temporal evaluation | In progress |
| MLflow experiment tracking | In progress |
| Single-command pipeline entry point | In progress |

## Planned enhancements — V2 (Final)

- **Serving** — FastAPI `/predict`, `/health`, `/metrics` in Docker, with the feature
  module shared between training and serving to guarantee parity.
- **CI/CD** — GitHub Actions running lint, tests, the data contract and an offline
  evaluation gate that blocks promotion on MAE regression.
- **Model registry** — MLflow Registry with explicit `None → Staging → Production`
  transitions as the source of truth.
- **Orchestration** — Prefect flows for `ingest → validate → train → evaluate → register`.
- **Drift monitoring** — Evidently computing PSI, KS and Jensen-Shannon divergence
  against a frozen reference window, exported to Prometheus and visualised in Grafana.
- **Detector validation** — confirm the monitor fires on the March–April 2020 shift
  within one batch at PSI > 0.2, with zero false alarms across the eight preceding
  stable months, and measure its sensitivity floor against SDV-injected drift of known
  magnitude.
- **Governance** — architecture diagram, `STACK.md` justifying each tool against a named
  alternative, and a rehearsed rollback procedure.

## Licence

Code is MIT licensed. The ACN-Data dataset is **not** covered by that licence and remains
subject to the publisher's terms; it is not redistributed here.
