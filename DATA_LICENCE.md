# ACN-Data terms and project data-sharing decision

The ACN-Data records used for this project are published by Caltech, in collaboration
with PowerFlex Systems, as an open dataset for research. The MIT licence in this
repository covers the code only; this notice does not grant a licence to the dataset.

## What the publisher's terms say

Checked on <https://ev.caltech.edu> on 2026-10-20:

- **Use:** registering for an API token means agreeing "to use ACN-Data only for
  educational and research purposes" (<https://ev.caltech.edu/register>).
- **Citation:** published work using ACN-Data is asked to cite the paper below
  (<https://ev.caltech.edu/dataset>, Citing tab).
- **Registration:** the publisher asks every user to register "to keep track of how much
  the dataset is being used and protect it from bad actors".
- **Redistribution:** not addressed. The site publishes no dataset licence and does not
  explicitly permit or forbid sharing the records.

## How this project shares the data

The capstone is submitted as this public GitHub repository only; no separate data
upload or ZIP accompanies it.

| Copy | Real ACN-Data records |
|---|---|
| Public GitHub repository (the submission) | **Not included.** Only data-free manifests (`data/raw/manifest.json` and `data/reference/manifest.json`: counts and SHA-256 hashes), retrieval code, a synthetic fixture, and aggregate metrics, figures and reports are published. |
| Local working copy of the author | The extract retrieved through the API, kept outside version control (`data/raw/*.jsonl` and `data/reference/*.parquet` are gitignored). |

The repository stays data-free because publishing the records there would let anyone
download them without registering, bypassing the publisher's usage tracking and the
educational-use agreement made at registration. Public republication would need written
permission from the dataset maintainer.

## How evaluators reproduce the licensed results

1. Register for a free API token at <https://ev.caltech.edu/register> and put it in
   `.env` as `ACN_API_TOKEN` (copy `.env.example`).
2. Run `python -m evcharge.pipeline --source raw --fetch` (see the README) to download
   the same sites and date range directly from the publisher and run the pipeline.
3. Run the raw pipeline and compare the recomputed `data_sha256`, split row counts and
   metrics with `data/raw/manifest.json`, `benchmarks/licensed_baseline.json` and
   `benchmarks/dwell_baseline.json`.

Without a token, the synthetic fixture exercises every stage of the same code path; its
metrics are not comparable with the licensed benchmarks.

## Disclosure: identifiers in early git history

A history audit before submission found that commits from `84c9d0f` (V1 baseline) up to
the Phase 8 commit used identifier values of one real ACN session as unit-test inputs in
`tests/test_pii_audit.py`: one `_id`, one `sessionID`, one `stationID`, two space labels
and two pseudonymous `userID` values. No energy, timing, location or other measurement
fields were included. The current tree replaces them with synthetic values of the same
shape. History was not rewritten because the values are pseudonymous, carry no
measurements, and a force-push would invalidate published commit references. Every other
blob ever committed was scanned against the 66,745 licensed session, object and user
identifiers with zero matches; the synthetic fixture has never overlapped the licensed
records.

## Conditions for anyone using the data

- Use it only for educational and research purposes, as agreed at registration.
- Do not republish or redistribute the records.
- Register at <https://ev.caltech.edu/register> and download the data directly from the
  publisher.
- Cite the paper below in any work that uses it.

## Required citation

Zachary J. Lee, Tongxin Li and Steven H. Low. "ACN-Data: Analysis and Applications of
an Open EV Charging Dataset." *Proceedings of the Tenth ACM International Conference on
Future Energy Systems (e-Energy '19)*, 2019.

Dataset information and access: <https://ev.caltech.edu/dataset>
