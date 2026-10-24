"""Turn the processed demand table into per-unit time series usable by both
the forecasting models and the simulation/RL environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DemandDataset:
    def __init__(self, demand_df: pd.DataFrame):
        self.df = demand_df.sort_values(["unit_id", "timestamp"]).reset_index(drop=True)
        self.unit_ids = sorted(self.df["unit_id"].unique())
        self.unit_names = self.df.drop_duplicates("unit_id").set_index("unit_id")["unit_name"].to_dict()
        self.elder_care_flags = (
            self.df.drop_duplicates("unit_id").set_index("unit_id")["is_elder_care"].to_dict()
        )
        self.timestamps = sorted(self.df["timestamp"].unique())

        self._series: dict[int, np.ndarray] = {}
        self._fares: dict[int, np.ndarray] = {}
        for uid in self.unit_ids:
            unit_df = self.df[self.df["unit_id"] == uid]
            self._series[uid] = unit_df["trip_count"].to_numpy(dtype=float)
            self._fares[uid] = unit_df["avg_fare"].to_numpy(dtype=float)

    @property
    def num_units(self) -> int:
        return len(self.unit_ids)

    @property
    def num_steps(self) -> int:
        return len(self.timestamps)

    def series(self, unit_id: int) -> np.ndarray:
        return self._series[unit_id]

    def avg_fare(self, unit_id: int, step: int) -> float:
        fares = self._fares[unit_id]
        return float(fares[min(step, len(fares) - 1)])

    def elder_care_mask(self) -> np.ndarray:
        return np.array([self.elder_care_flags[uid] for uid in self.unit_ids], dtype=bool)

    def train_test_split(self, train_frac: float) -> tuple["DemandDataset", "DemandDataset"]:
        split_idx = int(len(self.timestamps) * train_frac)
        split_ts = self.timestamps[split_idx]
        train_df = self.df[self.df["timestamp"] < split_ts]
        test_df = self.df[self.df["timestamp"] >= split_ts]
        return DemandDataset(train_df), DemandDataset(test_df)

    def as_matrix(self) -> np.ndarray:
        """Return demand as a (num_steps, num_units) matrix, aligned by timestamp."""
        pivot = self.df.pivot(index="timestamp", columns="unit_id", values="trip_count")
        pivot = pivot.reindex(columns=self.unit_ids).fillna(0.0)
        return pivot.to_numpy()
