"""Gymnasium environment simulating AV fleet allocation across city zones.

Each step, the agent outputs a preference score per zone. We turn that into a
target fleet distribution (softmax * total_vehicles), charge a reposition
cost for moving vehicles between zones, serve as much of the *actual*
realized demand as the allocated fleet allows, and collect fare revenue.
Zones flagged as elder-care must receive at least `min_service_level` of
their forecasted demand (or `min_vehicles`, whichever is higher); shortfalls
are penalized in the reward, giving the agent a strong incentive to satisfy
the constraint rather than chase profit alone.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.forecasting.dataset import DemandDataset
from src.forecasting.models import BaseForecaster


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class AVFleetAllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset: DemandDataset, forecaster: BaseForecaster, cfg: dict):
        super().__init__()
        self.dataset = dataset
        self.forecaster = forecaster
        self.n = dataset.num_units
        self.elder_care_mask = dataset.elder_care_mask()

        fleet_cfg, ec_cfg, rl_cfg = cfg["fleet"], cfg["elder_care"], cfg["rl"]
        self.total_vehicles = float(fleet_cfg["total_vehicles"])
        self.reposition_cost_per_vehicle = float(fleet_cfg["reposition_cost_per_vehicle"])
        self.revenue_multiplier = float(fleet_cfg["revenue_per_trip_multiplier"])
        self.min_service_level = float(ec_cfg["min_service_level"])
        self.min_vehicles = float(ec_cfg["min_vehicles"])
        self.penalty_weight = float(rl_cfg["elder_care_penalty_weight"])
        self.episode_length = int(rl_cfg["episode_length_steps"])

        self.demand_matrix = dataset.as_matrix()  # (num_steps, num_units)
        self.max_start = self.demand_matrix.shape[0] - self.episode_length - 1
        if self.max_start < 1:
            raise ValueError("Not enough time steps in dataset for one episode; reduce episode_length_steps")

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n,), dtype=np.float32)
        obs_dim = 3 * self.n + 2
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.fleet = np.zeros(self.n, dtype=np.float64)
        self.step_idx = 0
        self.t = 0

    def _forecast_all(self, t: int) -> np.ndarray:
        preds = np.zeros(self.n, dtype=np.float64)
        for i, uid in enumerate(self.dataset.unit_ids):
            recent = self.demand_matrix[max(0, t - 200):t, i]
            preds[i] = self.forecaster.predict_next(uid, recent) if len(recent) else 0.0
        return preds

    def _observation(self) -> np.ndarray:
        forecast = self._forecast_all(self.t)
        hour = self.t % 24
        hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        obs = np.concatenate([
            self.fleet / self.total_vehicles,
            forecast / max(forecast.max(), 1.0),
            self.elder_care_mask.astype(np.float64),
            [hour_sin, hour_cos],
        ])
        return obs.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = int(self.np_random.integers(0, self.max_start))
        self.step_idx = 0
        self.fleet = np.full(self.n, self.total_vehicles / self.n, dtype=np.float64)
        return self._observation(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        target_fractions = _softmax(action)
        target_fleet = target_fractions * self.total_vehicles

        forecast = self._forecast_all(self.t)
        required = np.maximum(forecast * self.min_service_level, 0.0)
        required = np.where(self.elder_care_mask, np.maximum(required, self.min_vehicles), 0.0)
        shortfall = np.maximum(required - target_fleet, 0.0)
        # Normalize by required count so the penalty is a bounded [0,1]-per-zone
        # shortfall fraction rather than a raw vehicle-count gap: with a fleet
        # far smaller than a zone's demand, an absolute penalty would dwarf
        # revenue regardless of policy and give PPO no learnable gradient.
        relative_shortfall = np.divide(shortfall, required, out=np.zeros_like(shortfall), where=required > 0)
        elder_care_penalty = float(relative_shortfall.sum()) * self.penalty_weight
        elder_care_violations = int(np.sum(shortfall > 1e-6))

        moved = np.abs(target_fleet - self.fleet).sum() / 2.0
        reposition_cost = moved * self.reposition_cost_per_vehicle

        self.fleet = target_fleet
        actual_demand = self.demand_matrix[self.t]
        served = np.minimum(self.fleet, actual_demand)

        fares = np.array([self.dataset.avg_fare(uid, self.t) for uid in self.dataset.unit_ids])
        revenue = float((served * fares).sum()) * self.revenue_multiplier

        profit = revenue - reposition_cost - elder_care_penalty
        reward = profit / 1000.0  # scale for stable RL training

        self.t += 1
        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.episode_length

        info = {
            "profit": profit,
            "revenue": revenue,
            "reposition_cost": reposition_cost,
            "elder_care_penalty": elder_care_penalty,
            "elder_care_violations": elder_care_violations,
            "served_demand": float(served.sum()),
            "actual_demand": float(actual_demand.sum()),
            # Per-zone breakdown (unit_id order), for schedule export/inspection.
            "zone_target_fleet": target_fleet.copy(),
            "zone_forecast": forecast.copy(),
            "zone_actual_demand": actual_demand.copy(),
            "zone_served": served.copy(),
        }
        return self._observation(), float(reward), terminated, truncated, info
