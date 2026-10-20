# Drift hypotheses (pre-detector)

These hypotheses and thresholds are registered before detector evaluation. They
are project hypotheses, not results; the monitoring work must report misses and
false alarms without retuning them on the evaluation batches.

The frozen reference is the half-open local-time interval
**2019-07-01 through 2020-02-29** (`2019-07-01 <= connectionTimeLocal <
2020-03-01`). Its data is stored as `data/reference/drift_reference_v1.parquet`
in the private evaluation copy, with row counts and SHA-256 values in the
data-free `data/reference/manifest.json`.

| ID | Hypothesis | Comparison and predeclared signal |
|---|---|---|
| H1 — pandemic regime | The March–April 2020 workplace shutdown changes session volume and delivered-energy behavior. | Compare each site's monthly count with the median monthly count in the reference; expect at least a 70% decrease at sites 0001 and 0002. Test `kWhDelivered` PSI against the frozen reference; the alert target is PSI > 0.2 within one monthly batch. |
| H2 — fleet mix | The multi-year mix shifts toward vehicles requesting larger energy replenishments, consistent with (but not proof of) larger battery capacity. | For sites present in both periods, compare the 90th percentile of positive `kWhRequested` in May–September 2021 with the reference. A rise of at least 20% is the declared signal. `kWhRequested` is only a proxy; ACN-Data does not provide battery capacity, so do not describe this as a direct battery measurement. |
| H3 — declared/delivered gap | Driver-declared demand becomes less representative of energy delivered. | On rows with positive `kWhRequested`, monitor `kWhRequested - kWhDelivered`. A reference-relative PSI > 0.2 or a median absolute change of at least 2 kWh is the declared signal; report both the sign and magnitude rather than assuming the direction. |

## Evaluation rules

- Freeze the snapshot and this document before running the drift detector.
- Compare like-for-like sites and monthly batches; report sample counts and missing
  declared-request coverage with every score.
- Use PSI > 0.2 as the proposal's alert target for H1, not as a universal statistical
  guarantee. Report KS and Jensen–Shannon alongside it in the later detector study.
- H2 and H3 are hypotheses to evaluate, not detector acceptance criteria. Their
  thresholds must not be revised after seeing their scores.
- The pre-shift false-alarm study must be out-of-sample: the 8 months that make up the
  reference cannot also be scored against that same reference and counted as independent
  evidence. Use leave-one-month-out references or an earlier window.
