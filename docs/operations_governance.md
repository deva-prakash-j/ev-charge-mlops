# Operations and governance evidence

## Model consumers and ownership

The accepted proposal identifies these consumers:

- Charging operations / site energy management.
- Charging app and driver-experience team.
- Smart-charging scheduler.
- Data governance and audit.

The MLOps engineer owns deployment, model promotion and rollback, monitoring, alert
response, and retraining operations. Charging operations accepts or rejects retrained
candidates before promotion.

For this academic demo, **Deva Prakash J** is the acting MLOps owner and first alert
responder. This is a prototype assignment, not a staffed service commitment. Before any
real production deployment, assign an on-call rota with primary/backup responders and
name the charging-operations approver. The demo does not have an always-on alert
webhook or a production response-time commitment.

## Rollback runbook

1. Identify the known-good Production model version in MLflow and the matching
   known-good API image tag/digest.
2. Set the MLflow `production` alias to that model version.
3. Set `EVCHARGE_IMAGE` to the known-good image tag and recreate the MLflow, bootstrap,
   and API containers together so they use the same application image.
4. Check `/health` for the expected model version and send a smoke request to
   `/predict`.
5. Inspect API health, errors, and latency. Do not expect input-drift measurements to
   return to reference levels solely because a model alias was rolled back.
6. Record the model version, image digest, smoke-test result, and operator/time.

The Compose default remains `evcharge-capstone:local`. `EVCHARGE_IMAGE` selects an
existing image tag for the shared application image without changing the MLflow
Production alias or persistent data volumes. For a normal rollback, use the existing
registry and prediction-log volumes; do not run `down -v`.

```powershell
$env:EVCHARGE_IMAGE = 'evcharge-capstone:<known-good-tag>'
docker compose up -d --no-build --force-recreate mlflow bootstrap api
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType 'application/json' -InFile .\examples\predict_request.json
Remove-Item Env:EVCHARGE_IMAGE
```

## Executed rollback drill

**Date:** 2026-10-20

**Environment:** isolated Docker Compose project `evcharge-phase7-rollback`; API host
port 8002 and MLflow host port 5002. The existing user-facing Compose project and MLflow
volume were not modified.

| Checkpoint | Registry Production alias | API image | Health | Prediction |
|---|---|---|---|---|
| Initial known-good | `evcharge-sample-energy` v1 | `sha256:ce43b1d9818f3c625ee8c504a0b4b7698f2f6028f5496a5d2f828a8277481891` | `status=ok`, version 1 | 10.490037 kWh, version 1 |
| Candidate promoted | v2 | same image digest | `status=ok`, version 2 | 10.490037 kWh, version 2 |
| Rollback complete | v1 | same previous image digest | `status=ok`, version 1 | 10.490037 kWh, version 1 |

The isolated registry created version 2 from the same run/artifact as version 1, copied
the feature-spec tags, switched the Production alias to v2, and confirmed the API loaded
and served it. The alias was then restored to v1 and the API container was recreated
using the `evcharge-capstone:phase7-previous` tag. The running container's image ID
matched the captured previous digest above. Health and prediction both returned version
1 afterward.

This drill verifies the alias-switch and previous-image redeploy mechanics. Because the
candidate reused the same model artifact, it is **not** a test of model quality, a
distinct image rollback, or recovery from a real production incident. The temporary
Compose containers, network, and volumes were removed after verification. The active
Compose API remained healthy on port 8001 and continued serving version 1.

## Licensing and redistribution boundary

The direct runtime and named tool license inventory is in [`../STACK.md`](../STACK.md).
It is not a full transitive SBOM. The project code's MIT license does not license the
ACN-Data extract; its educational/research terms and citation requirements are recorded
separately in [`../DATA_LICENCE.md`](../DATA_LICENCE.md).
