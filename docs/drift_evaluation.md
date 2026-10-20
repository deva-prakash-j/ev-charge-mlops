# Drift detector and retraining evaluation

The detector thresholds and hypotheses were declared in
[`drift_hypotheses.md`](drift_hypotheses.md) before scoring. The licensed snapshot
remains local to the evaluation copy; the tracked manifest identifies it by SHA-256
`fe8bbb47323dec1b88580bbe4a0863f03c2866728fd201f0e1ef29fc133d43eb` (19,144 rows).
This document contains aggregate measurements only.

## Reproduction

Run these commands in the licensed evaluation copy after normalising the raw extract,
creating the frozen reference, and registering a raw Production energy model:

```powershell
.\.venv\Scripts\python.exe -m evcharge.pipeline --source raw
.\.venv\Scripts\python.exe -m evcharge.validation.reference --source raw
.\.venv\Scripts\python.exe scripts\evaluate_drift.py
.\.venv\Scripts\python.exe scripts\evaluate_retraining_recovery.py
```

`evaluate_drift.py` writes the monthly target scores, leave-one-month-out results,
injected-shift sensitivity, and H2/H3 results to
`artifacts/drift/detector_evaluation.json`. `evaluate_retraining_recovery.py` performs
chronological training without registering or promoting candidates and writes
`artifacts/drift/retraining_recovery.json`. Both outputs are data-free aggregate reports
and are not committed.

## H1 — March–April 2020 regime change

The comparison uses the frozen 2019-07..2020-02 reference and the raw month across all
sites. The alert criterion is the predeclared delivered-energy PSI > 0.2.

| Month | Rows | Site 0001 vs reference monthly median | Site 0002 vs reference monthly median | `kWhDelivered` PSI | KS statistic (p-value) | JS distance | Alert |
|---|---:|---:|---:|---:|---:|---:|---|
| 2020-03 | 1,264 | 835 (−42.4%) | 381 (−53.8%) | 0.0132 | 0.0275 (0.327) | 0.0410 | No |
| 2020-04 | 240 | 201 (−86.1%) | 24 (−97.1%) | 0.2311 | 0.1800 (3.45e−7) | 0.1637 | **Yes** |

The detector did not alert on March delivered energy; it crossed the declared PSI
threshold in the April batch, one month later. April has no missing delivered-energy
values in the scored sample (240 current and 19,144 reference rows). This supports the
one-month detection target, but it is not an early March warning.

## Stable-period false alarms

Each reference month was compared with the other seven reference months (leave-one-month-
out), so no month was compared with a reference containing itself. At the target PSI
threshold, the result was **0 false alarms across 8 months**. Delivered-energy PSI ranged
from 0.0054 to 0.0217; all eight comparisons had complete target coverage. The
leave-one-month-out KS p-values also remained above 0.01 (the minimum was 0.0101).

## Controlled sensitivity

The reference `kWhDelivered` values were multiplied by a known factor while the other
features were held fixed. This direct transformation avoids SDV and does not assert a
causal or physically realistic intervention.

| Injected energy change | PSI | Alert at > 0.2 |
|---:|---:|---|
| 0% | 0.0000 | No |
| +10% | 0.0366 | No |
| +20% | 0.1117 | No |
| +30% | 0.2090 | **Yes** |
| +50% | 0.3104 | **Yes** |
| +100% | 0.7889 | **Yes** |

The first tested transformation to alert was +30%; this is a coarse sensitivity study,
not a universal detection limit.

## Other predeclared hypotheses

H2 compares the 90th percentile of positive requested energy by shared site between the
reference and 2021-05..2021-09. Site 0002 increased by 57.5%, while site 0001 was
essentially unchanged (−0.4%). Site 0019 increased by 185.6%, but its reference
comparison contains only 45 positive-request rows. This is mixed evidence for a shift in
requested-energy mix, not evidence of vehicle battery capacity.

H3 compares `kWhRequested - kWhDelivered` for positive-request rows between the same
periods. The median gap increased from 6.345 to 10.120 kWh (absolute median change
3.775 kWh); PSI was 0.2614, KS statistic 0.1655 (p=2.84e−98), and JS distance 0.1415.
There were 16,796 reference and 5,396 later positive-request rows, with complete gap
coverage among the scored rows. The declared gap-shift hypothesis is supported for this
comparison window.

## Alert and observability path

The monthly Prefect flow scores new/changed batches, exports per-feature PSI, KS,
Jensen–Shannon, sample counts, missingness, and alert state to Pushgateway, which
Prometheus scrapes. Prometheus rules cover delivered-energy PSI > 0.2, other PSI alerts,
and API scrape availability. Grafana provisions the Prometheus datasource and the
`EV Charge Drift Monitoring` dashboard.

The Prefect flow invokes the quality-gated retraining hook directly from the latest
scored result; the Prometheus alert is an operator-facing signal, not a webhook consumed
by Prefect. A validation run exported the April target score and Prometheus returned
PSI=0.2311. Batch-validation, score-retry, and one-retrain-per-fingerprint behavior are
covered by orchestration tests.

## Retraining recovery — acceptance target not met

The pre-shift baseline is the licensed HGB validation MAE, 4.3993 kWh. The declared
recovery ceiling is therefore 4.8393 kWh. In each experiment, training, validation, and
post-shift test periods are chronological; the later test period is excluded from model
fitting and candidate selection.

| Training through | Validation | Held-out post-shift test | Selected model | Test rows | MAE | Within 4.8393? |
|---|---|---|---|---:|---:|---|
| 2020-06 | 2020-07..2020-08 | 2020-09..2020-12 | HistGradientBoosting | 869 | 6.6406 | **No** |
| 2021-02 | 2021-03 | 2021-04..2021-05 | HistGradientBoosting | 2,055 | 5.7115 | **No** |

The best tested later-window result is still 29.8% above the pre-shift MAE, so the
proposal's 10% recovery target remains unmet. The detector, false-alarm study, sensitivity
study, monitoring stack, and guarded retraining path are implemented, but these results
do **not** demonstrate model recovery. Do not promote the recovery criterion to a pass;
further model or feature investigation is required.
