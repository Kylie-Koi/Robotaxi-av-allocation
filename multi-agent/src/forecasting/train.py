"""Train the configured demand forecaster on the processed demand table and
save it to models/forecaster.pkl (or models/lstm_forecaster.pt for the LSTM).
"""

import pickle

import pandas as pd
import torch

from src.forecasting.dataset import DemandDataset
from src.forecasting.models import LSTMForecaster, build_forecaster
from src.utils.config import load_config, resolve_path


def train_forecaster(cfg: dict):
    processed_dir = resolve_path(cfg, cfg["data"]["processed_dir"])
    demand_df = pd.read_parquet(processed_dir / "demand.parquet")
    dataset = DemandDataset(demand_df)

    train_ds, test_ds = dataset.train_test_split(cfg["forecasting"]["train_frac"])
    series_by_unit = {uid: train_ds.series(uid) for uid in train_ds.unit_ids}

    forecaster = build_forecaster(cfg)
    forecaster.fit(series_by_unit)

    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(forecaster, LSTMForecaster):
        torch.save(forecaster.model.state_dict(), models_dir / "lstm_forecaster.pt")
        with open(models_dir / "lstm_forecaster_meta.pkl", "wb") as f:
            pickle.dump({"scale": forecaster._scale, "lookback_steps": forecaster.lookback_steps}, f)
        print(f"[saved] LSTM forecaster -> {models_dir}")
    else:
        with open(models_dir / "forecaster.pkl", "wb") as f:
            pickle.dump(forecaster, f)
        print(f"[saved] forecaster -> {models_dir / 'forecaster.pkl'}")

    return forecaster, train_ds, test_ds


if __name__ == "__main__":
    train_forecaster(load_config())
