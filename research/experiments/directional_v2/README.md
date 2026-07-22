# Directional V2 Forward Experiment

This experiment is isolated from the live strategy and from automatic model promotion.

The old 90-day OOS interval is retired evidence. It may explain why the old strategy was eliminated, but it must not select any directional-v2 label, signal, model, or gate parameter.

The final holdout starts at `2026-07-23T00:00:00Z`. Before `2026-08-22T00:00:00Z`, or before 30 closed trades exist, the only valid final result is `WATCH`.

Changing `spec.json` after the holdout starts invalidates this experiment. Create a new experiment id and restart the holdout clock instead.
