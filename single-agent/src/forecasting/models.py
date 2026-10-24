"""Demand forecasting models, all sharing a common fit/predict interface so the
simulation/RL code can swap between a cheap baseline and a learned model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BaseForecaster:
    def fit(self, series_by_unit: dict[int, np.ndarray]) -> None:
        raise NotImplementedError

    def predict_next(self, unit_id: int, recent_values: np.ndarray) -> float:
        """Given the most recent `lookback_steps` observed values for a unit,
        predict demand (trip_count) for the next time step."""
        raise NotImplementedError


class SeasonalNaiveForecaster(BaseForecaster):
    """Predicts next value = average of the same time-of-week slot in history.

    Cheap, robust baseline: captures daily/weekly seasonality without training,
    and is what the RL env falls back to when no learned model is provided.
    """

    def __init__(self, period: int = 168):
        self.period = period
        self.history: dict[int, np.ndarray] = {}

    def fit(self, series_by_unit: dict[int, np.ndarray]) -> None:
        self.history = series_by_unit

    def predict_next(self, unit_id: int, recent_values: np.ndarray) -> float:
        if len(recent_values) >= self.period:
            return float(recent_values[-self.period])
        return float(np.mean(recent_values)) if len(recent_values) else 0.0


class LSTMDemandModel(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    """Learned forecaster: one shared LSTM across all units, conditioned on
    each unit's own recent window (per-unit models would overfit with this
    much data; a shared model generalizes across units).
    """

    def __init__(self, lookback_steps: int, hidden_size: int = 64, num_layers: int = 2,
                 epochs: int = 20, lr: float = 1e-3, batch_size: int = 64):
        self.lookback_steps = lookback_steps
        self.model = LSTMDemandModel(hidden_size, num_layers)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._scale: dict[int, float] = {}

    def _make_windows(self, series_by_unit: dict[int, np.ndarray]):
        X, y = [], []
        for unit_id, series in series_by_unit.items():
            scale = max(series.max(), 1.0)
            self._scale[unit_id] = scale
            norm = series / scale
            for i in range(len(norm) - self.lookback_steps):
                X.append(norm[i:i + self.lookback_steps])
                y.append(norm[i + self.lookback_steps])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, series_by_unit: dict[int, np.ndarray]) -> None:
        X, y = self._make_windows(series_by_unit)
        X_t = torch.tensor(X).unsqueeze(-1)
        y_t = torch.tensor(y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n = len(X_t)

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                optimizer.zero_grad()
                pred = self.model(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            print(f"[lstm] epoch {epoch + 1}/{self.epochs} loss={epoch_loss / n:.4f}")

    def predict_next(self, unit_id: int, recent_values: np.ndarray) -> float:
        scale = self._scale.get(unit_id, max(recent_values.max(), 1.0) if len(recent_values) else 1.0)
        window = recent_values[-self.lookback_steps:] / scale
        if len(window) < self.lookback_steps:
            window = np.pad(window, (self.lookback_steps - len(window), 0))
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(window, dtype=torch.float32).reshape(1, -1, 1)
            pred = self.model(x).item()
        return max(pred * scale, 0.0)


def build_forecaster(cfg: dict) -> BaseForecaster:
    fcfg = cfg["forecasting"]
    if fcfg["model"] == "seasonal_naive":
        return SeasonalNaiveForecaster(period=fcfg["lookback_steps"])
    if fcfg["model"] == "lstm":
        lstm_cfg = fcfg["lstm"]
        return LSTMForecaster(
            lookback_steps=fcfg["lookback_steps"],
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            epochs=lstm_cfg["epochs"],
            lr=lstm_cfg["lr"],
            batch_size=lstm_cfg["batch_size"],
        )
    raise ValueError(f"Unknown forecasting model: {fcfg['model']}")
