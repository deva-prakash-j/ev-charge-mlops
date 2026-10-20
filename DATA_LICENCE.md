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

| Copy | Real ACN-Data records |
|---|---|
| Public GitHub repository | **Not included.** Only data-free manifests (`data/raw/manifest.json` and `data/reference/manifest.json`: counts and hashes), retrieval code and a synthetic fixture are published. |
| upGrad submission (ZIP / platform upload) | **Included unmodified** in `data/raw/`, plus the versioned derived drift reference in `data/reference/`, solely so evaluators can reproduce the results without registering. |

The evaluation copy is shared for educational assessment only, which is within the
publisher's stated purpose and matches the accepted proposal ("no redistribution beyond
the … snapshot submitted to upGrad solely for evaluation"). It ships the extract as
retrieved rather than only a derived snapshot, so the pipeline can be reproduced from its
first stage. The drift reference is derived from the 2019-07 through 2020-02 window and
omits session and user identifiers. The original extract keeps the publisher's
pseudonymous `userID`.

The public repository stays data-free because publishing there would let anyone download
the records without registering, bypassing the publisher's usage tracking and the
educational-use agreement made at registration. Public republication would need written
permission from the dataset maintainer.

## Conditions for recipients of the evaluation copy

- Use it only to evaluate this capstone.
- Do not republish or redistribute it.
- For any other use, register at <https://ev.caltech.edu/register> and download the data
  directly from the publisher.
- Cite the paper below in any work that uses it.

## Required citation

Zachary J. Lee, Tongxin Li and Steven H. Low. "ACN-Data: Analysis and Applications of
an Open EV Charging Dataset." *Proceedings of the Tenth ACM International Conference on
Future Energy Systems (e-Energy '19)*, 2019.

Dataset information and access: <https://ev.caltech.edu/dataset>
