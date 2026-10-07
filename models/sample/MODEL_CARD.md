# Model card - hist_gradient_boosting (sample)

- Target: kWhDelivered (kWh delivered in a charging session)
- Trained: 2026-10-07T13:08:49+00:00
- Git commit: 83beb1b53eb8
- Features: 16
- Validation MAE: 6.3419
- Post-shift MAE: 7.2047

Inputs are plug-in-time only. Post-hoc columns (disconnectTime, doneChargingTime) are rejected by the feature builder.

Load with:

```python
import joblib
from evcharge.features.build import FeatureSpec
model = joblib.load('models/sample/model.joblib')
spec = FeatureSpec.load('models/sample/feature_spec.json')
```
