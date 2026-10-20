# Plug-in-time feature specification

Both the delivered-energy model (`kWhDelivered`) and connected-duration model
(`connectedHours`) use the same information available when a vehicle plugs in.
The implementation is the shared builder in `src/evcharge/features/build.py`,
configured by `configs/features.yaml`. The fitted category vocabulary and feature
order are stored beside each estimator.

## Included features and transformations

| Inputs / transformation | Output feature(s) | Reason |
|---|---|---|
| Local `connectionTimeLocal` | `arrival_hour`, `arrival_dayofweek`, `arrival_is_weekend`, `arrival_month` | Arrival patterns differ by time, weekday/weekend and season; local time matches site operations. |
| Arrival hour | `arrival_hour_sin`, `arrival_hour_cos` | Cyclical encoding makes 23:00 adjacent to 00:00 instead of treating them as opposite ends of a linear scale. |
| `siteID`, `clusterID`, `stationID` | `siteID_code`, `clusterID_code`, `stationID_code` | Capture site, equipment grouping and station effects. Identifiers are normalized exactly as during ingest; unseen categories map to the fitted unknown code. |
| `kWhRequested`, `milesRequested`, `minutesAvailable`, `WhPerMile` | Same four numeric columns, coerced to float | Driver-declared demand and availability are known at plug-in and informative for both outcomes. Missing requests remain missing for the estimator to handle. |
| `requestedDeparture - connectionTime` | `declared_duration_minutes` | Encodes the driver's requested dwell window using only the original request and arrival. |
| `kWhRequested / minutesAvailable` | `declared_kwh_per_minute` | Captures the requested energy rate; zero or missing availability yields missing rather than infinity. |
| Presence of `kWhRequested` | `has_user_inputs` | Separates a genuinely absent driver request from a numeric zero. |

The model has 16 features: six temporal, three declared-request derived, four numeric
request, and three encoded categorical features.

## Deliberately excluded

- **Connected duration (`connectedHours`)** is the target for the dwell model, never an
  input to either model. It is only known after disconnect.
- **Active-charging duration** requires `doneChargingTime`, which is post-session.
- **Idle-after-charge time** requires both `doneChargingTime` and `disconnectTime`.
- **Declared-versus-delivered energy gap** requires the post-session `kWhDelivered`.
  It is retained as a drift-monitoring signal in the frozen reference snapshot, not a
  prediction input.
- **`disconnectTime`, `doneChargingTime`, `kWhDelivered`, `userInputs_count`**, and
  publisher/person/session identifiers are prohibited by the data/feature leakage
  checks.
- **Rolling site occupancy is deferred.** It needs a point-in-time, as-of join to
  sessions already active at the request timestamp, including an explicit policy for
  concurrent arrivals and late events. Computing it from the batch being scored would
  make a prediction depend on batch composition. Do not add it until that stateful
  feature has a parity and leakage audit.
