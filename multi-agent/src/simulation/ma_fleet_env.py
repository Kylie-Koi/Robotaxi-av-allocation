"""Multi-agent fleet allocation environment: one agent per zone instead of a
single monolithic policy choosing all zones' allocations at once.

Design follows the "zone/grid as agent" pattern from Lin et al. (KDD 2018,
"Efficient Large-Scale Fleet Management via Multi-Agent Deep RL"), which
models each grid cell as an agent sharing one policy network across all
grids (parameter sharing), conditioned on its own state plus a city-wide
context vector so agents implicitly coordinate without needing a full
joint action space. We adapt that to our "allocate stationed vehicles"
framing (continuous target-fleet decisions, not discrete per-driver moves):

- Agent = one zone. All agents share one policy (PettingZoo + SuperSuit +
  SB3's standard parameter-sharing recipe, i.e. "independent PPO", shown by
  de Witt et al. 2020 to be a strong, simple MARL baseline).
- Observation = local state (own fleet share, own demand forecast, own
  elder-care flag) + city-wide context (aggregate fleet/demand pressure,
  time-of-day) -- the context vector plays the role of Lin et al.'s shared
  "global state" component appended to every agent's observation.
- Action = one scalar preference score per agent (same semantics as the
  single-agent project's per-zone logits).
- Joint feasibility: exactly as in the single-agent env, all agents'
  scalar actions are gathered and passed through one softmax so the fleet
  is allocated as fractions of `total_vehicles` -- this is the "coordination"
  step, analogous to Lin et al.'s centralized masking/normalization of the
  joint policy, just computed once per environment tick rather than
  per-agent.
- Reward: decomposed per agent as that zone's own (revenue - reposition
  cost - elder-care penalty), so credit assignment is local and the sum
  across agents reproduces the same global objective as the single-agent
  env (maximize profit, prioritize elder care) -- the target is unchanged,
  only the policy architecture is.
"""

from __future__ import annotations

import functools

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from src.forecasting.dataset import DemandDataset
from src.forecasting.models import BaseForecaster


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class AVFleetAllocationMAEnv(ParallelEnv):
    metadata = {"name": "av_fleet_allocation_ma_v0"}

    def __init__(self, dataset: DemandDataset, forecaster: BaseForecaster, cfg: dict):
        super().__init__()
        self.render_mode = None
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

        self.demand_matrix = dataset.as_matrix()
        self.max_start = self.demand_matrix.shape[0] - self.episode_length - 1
        if self.max_start < 1:
            raise ValueError("Not enough time steps for one episode; reduce episode_length_steps")

        self.possible_agents = [f"zone_{uid}" for uid in dataset.unit_ids]
        self.agents = list(self.possible_agents)
        self._uid_by_agent = dict(zip(self.possible_agents, dataset.unit_ids))
        self._idx_by_agent = {a: i for i, a in enumerate(self.possible_agents)}

        self.fleet = np.zeros(self.n, dtype=np.float64)
        self.t = 0
        self.step_idx = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # [own_fleet_frac, own_forecast_norm, own_elder_care_flag,
        #  city_fleet_pressure, city_demand_norm, hour_sin, hour_cos]
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

    def _forecast_all(self, t: int) -> np.ndarray:
        preds = np.zeros(self.n, dtype=np.float64)
        for i, uid in enumerate(self.dataset.unit_ids):
            recent = self.demand_matrix[max(0, t - 200):t, i]
            preds[i] = self.forecaster.predict_next(uid, recent) if len(recent) else 0.0
        return preds

    def _observations(self) -> dict:
        forecast = self._forecast_all(self.t)
        forecast_scale = max(forecast.max(), 1.0)
        hour = self.t % 24
        hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)

        # City-wide context shared by every agent (plays the role of Lin et
        # al.'s global state component appended to each per-grid agent obs).
        city_fleet_pressure = self.fleet.sum() / self.total_vehicles
        city_demand_norm = forecast.sum() / (self.n * forecast_scale)

        obs = {}
        for agent in self.agents:
            i = self._idx_by_agent[agent]
            obs[agent] = np.array([
                self.fleet[i] / self.total_vehicles,
                forecast[i] / forecast_scale,
                float(self.elder_care_mask[i]),
                city_fleet_pressure,
                city_demand_norm,
                hour_sin,
                hour_cos,
            ], dtype=np.float32)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = list(self.possible_agents)
        rng = np.random.default_rng(seed)
        self.t = int(rng.integers(0, self.max_start))
        self.step_idx = 0
        self.fleet = np.full(self.n, self.total_vehicles / self.n, dtype=np.float64)
        obs = self._observations()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(self, actions: dict):
        action_vec = np.array([actions[agent][0] for agent in self.agents], dtype=np.float64)
        target_fractions = _softmax(action_vec)
        target_fleet = target_fractions * self.total_vehicles

        forecast = self._forecast_all(self.t)
        required = np.maximum(forecast * self.min_service_level, 0.0)
        required = np.where(self.elder_care_mask, np.maximum(required, self.min_vehicles), 0.0)
        shortfall = np.maximum(required - target_fleet, 0.0)
        relative_shortfall = np.divide(shortfall, required, out=np.zeros_like(shortfall), where=required > 0)
        elder_care_penalty = relative_shortfall * self.penalty_weight  # per-zone, not summed

        moved = np.abs(target_fleet - self.fleet)
        reposition_cost = moved * self.reposition_cost_per_vehicle  # per-zone

        self.fleet = target_fleet
        actual_demand = self.demand_matrix[self.t]
        served = np.minimum(self.fleet, actual_demand)
        fares = np.array([self.dataset.avg_fare(uid, self.t) for uid in self.dataset.unit_ids])
        revenue = served * fares * self.revenue_multiplier  # per-zone

        profit_per_zone = revenue - reposition_cost - elder_care_penalty
        reward_per_zone = profit_per_zone / 1000.0

        self.t += 1
        self.step_idx += 1
        truncated = self.step_idx >= self.episode_length

        rewards, terminations, truncations, infos = {}, {}, {}, {}
        for agent in self.agents:
            i = self._idx_by_agent[agent]
            rewards[agent] = float(reward_per_zone[i])
            terminations[agent] = False
            truncations[agent] = truncated
            infos[agent] = {
                "profit": float(profit_per_zone[i]),
                "revenue": float(revenue[i]),
                "reposition_cost": float(reposition_cost[i]),
                "elder_care_penalty": float(elder_care_penalty[i]),
                "elder_care_violation": bool(shortfall[i] > 1e-6),
                "served_demand": float(served[i]),
                "actual_demand": float(actual_demand[i]),
                "target_fleet": float(target_fleet[i]),
                "forecast": float(forecast[i]),
            }

        obs = self._observations()
        return obs, rewards, terminations, truncations, infos
