"""Evaluate a trained forecaster on held-out data: MAE/RMSE per unit and overall,
plus a quick actual-vs-predicted plot for a sample unit.
"""

import numpy as np

from src.forecasting.dataset import DemandDataset
from src.forecasting.train import train_forecaster
from src.utils.config import load_config, resolve_path


def evaluate_forecaster(forecaster, test_ds: DemandDataset, lookback_steps: int) -> dict:
    errors = []
    per_unit_mae = {}

    for uid in test_ds.unit_ids:
        series = test_ds.series(uid)
        if len(series) <= lookback_steps:
            continue
        unit_errors = []
        for t in range(lookback_steps, len(series)):
            recent = series[:t]
            pred = forecaster.predict_next(uid, recent)
            actual = series[t]
            unit_errors.append(abs(pred - actual))
        if unit_errors:
            per_unit_mae[uid] = float(np.mean(unit_errors))
            errors.extend(unit_errors)

    return {
        "mae": float(np.mean(errors)) if errors else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan"),
        "per_unit_mae": per_unit_mae,
    }


def main():
    cfg = load_config()
    forecaster, _, test_ds = train_forecaster(cfg)
    metrics = evaluate_forecaster(forecaster, test_ds, cfg["forecasting"]["lookback_steps"])

    print(f"Overall MAE:  {metrics['mae']:.3f} trips/bin")
    print(f"Overall RMSE: {metrics['rmse']:.3f} trips/bin")
    for uid, mae in sorted(metrics["per_unit_mae"].items()):
        name = test_ds.unit_names.get(uid, uid)
        print(f"  unit {uid} ({name}): MAE={mae:.3f}")

    outputs_dir = resolve_path(cfg, cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    import json
    metrics_json = {**metrics, "per_unit_mae": {str(k): v for k, v in metrics["per_unit_mae"].items()}}
    with open(outputs_dir / "forecast_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)


if __name__ == "__main__":
    main()
