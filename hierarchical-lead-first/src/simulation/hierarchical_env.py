"""Hierarchical fleet allocation: one lead/manager agent sets a per-zone
"directive" each hour, and 7 worker agents (one per zone, shared policy)
each refine their own zone's directive before the allocation is finalized.

Follows Ahilan & Dayan (2019), "Feudal Multi-Agent Hierarchies for
Cooperative RL": one manager observing global state issues a goal per
worker; workers act simultaneously, conditioned on their own goal
component. Adapted to this domain: the lead's reward is the same global
profit/elder-care objective as the single-agent project (it IS the
system-level objective, not an abstract subgoal), while each worker's
reward is its own zone's local profit/elder-care outcome -- the actual
result of the combined (directive + refinement) action, not goal-adherence
per se, since here the "right" allocation is defined by the profit/elder-
care objective rather than a navigational target.

Because stable-baselines3 only trains single-policy (homogeneous) agents,
the lead and workers are trained as two SEPARATE PPO models in alternating
rounds (train_hierarchical.py): each round, the lead trains against the
CURRENT frozen worker policy, then the workers train against the newly
updated frozen lead policy, and repeat. Ahilan & Dayan's own ablations
found joint/co-located training of manager and workers outperforms
training them in isolation -- this alternation is this project's practical
approximation of that, achievable with off-the-shelf single-agent PPO.

Two environment classes share one underlying simulation:
- LeadEnv (gym.Env): the lead's view. Action = 7 directive scores.
  Internally calls a (possibly None) frozen worker model to get each
  zone's refinement, so with a worker model attached, stepping LeadEnv
  alone runs the FULL combined policy -- this is also what evaluate/export
  scripts use.
- WorkerEnv (PettingZoo ParallelEnv): the workers' view, one agent per
  zone. Internally calls a (possibly None) frozen lead model once per
  tick to get this tick's directive, which becomes part of each worker's
  observation.
"""

from __future__ import annotations

import functools

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.forecasting.dataset import DemandDataset
from src.forecasting.models import BaseForecaster


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class _SimCore:
    """Shared simulation state/physics used by both LeadEnv and WorkerEnv."""

    def _init_core(self, dataset: DemandDataset, forecaster: BaseForecaster, cfg: dict):
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

        self.fleet = np.zeros(self.n, dtype=np.float64)
        self.t = 0
        self.step_idx = 0

    def _forecast_all(self, t: int) -> np.ndarray:
        preds = np.zeros(self.n, dtype=np.float64)
        for i, uid in enumerate(self.dataset.unit_ids):
            recent = self.demand_matrix[max(0, t - 200):t, i]
            preds[i] = self.forecaster.predict_next(uid, recent) if len(recent) else 0.0
        return preds

    def _trip_rate_all(self, t: int) -> np.ndarray:
        """Trips one vehicle can complete per hour in each zone -- a vehicle
        tied up on a long trip can't also serve a second rider that hour."""
        durations = np.array([self.dataset.avg_trip_duration(uid, t) for uid in self.dataset.unit_ids])
        return 60.0 / durations

    def _reset_core(self, rng: np.random.Generator):
        self.t = int(rng.integers(0, self.max_start))
        self.step_idx = 0
        self.fleet = np.full(self.n, self.total_vehicles / self.n, dtype=np.float64)

    def _advance(self, final_logits: np.ndarray, forecast: np.ndarray) -> dict:
        """Apply a joint (directive + refinement) action, advance one tick.

        Returns a dict of per-zone arrays plus `truncated`. Does not reset
        -- callers bump self.t/self.step_idx themselves after reading state.
        """
        target_fractions = _softmax(final_logits)
        target_fleet = target_fractions * self.total_vehicles
        trip_rate = self._trip_rate_all(self.t)

        # Elder-care floor expressed as the vehicle count needed to deliver
        # the required demand coverage, given this zone's throughput.
        required_demand = np.maximum(forecast * self.min_service_level, 0.0)
        required = required_demand / trip_rate
        required = np.where(self.elder_care_mask, np.maximum(required, self.min_vehicles), 0.0)
        shortfall = np.maximum(required - target_fleet, 0.0)
        relative_shortfall = np.divide(shortfall, required, out=np.zeros_like(shortfall), where=required > 0)
        elder_care_penalty = relative_shortfall * self.penalty_weight

        moved = np.abs(target_fleet - self.fleet)
        reposition_cost = moved * self.reposition_cost_per_vehicle

        self.fleet = target_fleet
        actual_demand = self.demand_matrix[self.t]
        effective_capacity = self.fleet * trip_rate
        served = np.minimum(effective_capacity, actual_demand)
        fares = np.array([self.dataset.avg_fare(uid, self.t) for uid in self.dataset.unit_ids])
        revenue = served * fares * self.revenue_multiplier

        profit_per_zone = revenue - reposition_cost - elder_care_penalty

        self.t += 1
        self.step_idx += 1
        truncated = self.step_idx >= self.episode_length

        return {
            "target_fleet": target_fleet,
            "forecast": forecast,
            "actual_demand": actual_demand,
            "served": served,
            "revenue": revenue,
            "reposition_cost": reposition_cost,
            "elder_care_penalty": elder_care_penalty,
            "elder_care_violation": shortfall > 1e-6,
            "profit_per_zone": profit_per_zone,
            "truncated": truncated,
        }

    def _city_context(self, forecast: np.ndarray) -> tuple[float, float, float, float]:
        forecast_scale = max(forecast.max(), 1.0)
        hour = self.t % 24
        hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        city_fleet_pressure = self.fleet.sum() / self.total_vehicles
        city_demand_norm = forecast.sum() / (self.n * forecast_scale)
        return forecast_scale, city_fleet_pressure, city_demand_norm, (hour_sin, hour_cos)


class LeadEnv(gym.Env, _SimCore):
    """The lead/manager's view: observes the whole city, outputs one
    directive score per zone. With a worker model attached, this alone
    runs the full combined (lead + workers) policy."""

    def __init__(self, dataset: DemandDataset, forecaster: BaseForecaster, cfg: dict, worker_model=None):
        super().__init__()
        self._init_core(dataset, forecaster, cfg)
        self.worker_model = worker_model

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n,), dtype=np.float32)
        obs_dim = 4 * self.n + 2  # [fleet_frac, forecast_norm, elder_care, trip_rate_norm] per zone + hour_sin/cos
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _lead_observation(self, forecast: np.ndarray) -> np.ndarray:
        forecast_scale, _, _, (hour_sin, hour_cos) = self._city_context(forecast)
        trip_rate = self._trip_rate_all(self.t)
        obs = np.concatenate([
            self.fleet / self.total_vehicles,
            forecast / forecast_scale,
            self.elder_care_mask.astype(np.float64),
            trip_rate / max(trip_rate.max(), 1.0),
            [hour_sin, hour_cos],
        ])
        return obs.astype(np.float32)

    def _worker_refinements(self, lead_action: np.ndarray, forecast: np.ndarray) -> np.ndarray:
        if self.worker_model is None:
            return np.zeros(self.n, dtype=np.float64)
        forecast_scale, city_fleet_pressure, city_demand_norm, (hour_sin, hour_cos) = self._city_context(forecast)
        trip_rate = self._trip_rate_all(self.t)
        trip_rate_scale = max(trip_rate.max(), 1.0)
        worker_obs = np.stack([
            np.array([
                self.fleet[i] / self.total_vehicles,
                forecast[i] / forecast_scale,
                float(self.elder_care_mask[i]),
                float(lead_action[i]),
                trip_rate[i] / trip_rate_scale,
                city_fleet_pressure,
                city_demand_norm,
                hour_sin,
                hour_cos,
            ], dtype=np.float32)
            for i in range(self.n)
        ])
        adjustments, _ = self.worker_model.predict(worker_obs, deterministic=True)
        return adjustments.reshape(self.n).astype(np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_core(self.np_random)
        forecast = self._forecast_all(self.t)
        return self._lead_observation(forecast), {}

    def step(self, lead_action: np.ndarray):
        lead_action = np.asarray(lead_action, dtype=np.float64)
        forecast = self._forecast_all(self.t)
        adjustments = self._worker_refinements(lead_action, forecast)
        final_logits = lead_action + adjustments

        result = self._advance(final_logits, forecast)
        profit = float(result["profit_per_zone"].sum())
        reward = profit / 1000.0

        next_forecast = self._forecast_all(self.t) if not result["truncated"] else forecast
        obs = self._lead_observation(next_forecast)
        info = {
            "profit": profit,
            "revenue": float(result["revenue"].sum()),
            "reposition_cost": float(result["reposition_cost"].sum()),
            "elder_care_penalty": float(result["elder_care_penalty"].sum()),
            "elder_care_violations": int(result["elder_care_violation"].sum()),
            "served_demand": float(result["served"].sum()),
            "actual_demand": float(result["actual_demand"].sum()),
            "zone_target_fleet": result["target_fleet"],
            "zone_forecast": result["forecast"],
            "zone_actual_demand": result["actual_demand"],
            "zone_served": result["served"],
        }
        return obs, float(reward), False, result["truncated"], info


class WorkerEnv(ParallelEnv, _SimCore):
    """The workers' view: one agent per zone, conditioned on the (possibly
    frozen) lead's directive for that zone this tick."""

    metadata = {"name": "av_fleet_allocation_workers_v0"}

    def __init__(self, dataset: DemandDataset, forecaster: BaseForecaster, cfg: dict, lead_model_path: str | None = None):
        super().__init__()
        self.render_mode = None
        self._init_core(dataset, forecaster, cfg)
        # A path, not a live model object: SuperSuit's vectorization
        # cloudpickles this whole env (even at num_vec_envs=1), and SB3
        # model objects contain unpicklable internals (e.g. torch locks).
        # The model is lazily loaded from disk on first use instead, so
        # each (post-unpickle) copy loads its own instance.
        self.lead_model_path = lead_model_path
        self._lead_model = None

        self.possible_agents = [f"zone_{uid}" for uid in dataset.unit_ids]
        self.agents = list(self.possible_agents)
        self._idx_by_agent = {a: i for i, a in enumerate(self.possible_agents)}
        self._uid_by_agent = dict(zip(self.possible_agents, dataset.unit_ids))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

    def _get_lead_model(self):
        if self._lead_model is None and self.lead_model_path is not None:
            from stable_baselines3 import PPO
            self._lead_model = PPO.load(self.lead_model_path)
        return self._lead_model

    def _lead_directive(self, forecast: np.ndarray) -> np.ndarray:
        lead_model = self._get_lead_model()
        if lead_model is None:
            return np.zeros(self.n, dtype=np.float64)
        forecast_scale = max(forecast.max(), 1.0)
        hour = self.t % 24
        hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        trip_rate = self._trip_rate_all(self.t)
        lead_obs = np.concatenate([
            self.fleet / self.total_vehicles,
            forecast / forecast_scale,
            self.elder_care_mask.astype(np.float64),
            trip_rate / max(trip_rate.max(), 1.0),
            [hour_sin, hour_cos],
        ]).astype(np.float32)
        action, _ = lead_model.predict(lead_obs[None, :], deterministic=True)
        return action[0].astype(np.float64)

    def _observations(self, forecast: np.ndarray, lead_directive: np.ndarray) -> dict:
        forecast_scale, city_fleet_pressure, city_demand_norm, (hour_sin, hour_cos) = self._city_context(forecast)
        trip_rate = self._trip_rate_all(self.t)
        trip_rate_scale = max(trip_rate.max(), 1.0)
        obs = {}
        for agent in self.agents:
            i = self._idx_by_agent[agent]
            obs[agent] = np.array([
                self.fleet[i] / self.total_vehicles,
                forecast[i] / forecast_scale,
                float(self.elder_care_mask[i]),
                float(lead_directive[i]),
                trip_rate[i] / trip_rate_scale,
                city_fleet_pressure,
                city_demand_norm,
                hour_sin,
                hour_cos,
            ], dtype=np.float32)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = list(self.possible_agents)
        rng = np.random.default_rng(seed)
        self._reset_core(rng)
        forecast = self._forecast_all(self.t)
        lead_directive = self._lead_directive(forecast)
        obs = self._observations(forecast, lead_directive)
        return obs, {agent: {} for agent in self.agents}

    def step(self, actions: dict):
        forecast = self._forecast_all(self.t)
        lead_directive = self._lead_directive(forecast)
        worker_adjustments = np.array([actions[a][0] for a in self.agents], dtype=np.float64)
        final_logits = lead_directive + worker_adjustments

        result = self._advance(final_logits, forecast)

        rewards, terminations, truncations, infos = {}, {}, {}, {}
        for agent in self.agents:
            i = self._idx_by_agent[agent]
            rewards[agent] = float(result["profit_per_zone"][i]) / 1000.0
            terminations[agent] = False
            truncations[agent] = result["truncated"]
            infos[agent] = {
                "profit": float(result["profit_per_zone"][i]),
                "revenue": float(result["revenue"][i]),
                "reposition_cost": float(result["reposition_cost"][i]),
                "elder_care_penalty": float(result["elder_care_penalty"][i]),
                "elder_care_violation": bool(result["elder_care_violation"][i]),
                "served_demand": float(result["served"][i]),
                "actual_demand": float(result["actual_demand"][i]),
                "target_fleet": float(result["target_fleet"][i]),
                "forecast": float(result["forecast"][i]),
            }

        next_forecast = self._forecast_all(self.t) if not result["truncated"] else forecast
        next_directive = self._lead_directive(next_forecast) if not result["truncated"] else lead_directive
        obs = self._observations(next_forecast, next_directive)
        return obs, rewards, terminations, truncations, infos
