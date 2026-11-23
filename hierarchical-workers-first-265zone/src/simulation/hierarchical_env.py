"""Workers-first hierarchical fleet allocation: the inverse of the
lead-first hierarchical sibling project. Here, 7 worker agents (one per
zone, shared policy) act FIRST, each producing a raw per-zone action score.
All 7 scores are combined via the same joint-softmax feasibility step used
throughout this portfolio, producing a PROVISIONAL fleet allocation. A
single lead/coordinator agent then observes that provisional allocation
plus city-wide context and outputs a bounded per-zone corrective adjustment
SECOND, producing the FINAL allocation that actually determines reward.

Because stable-baselines3 only trains single-policy (homogeneous) agents,
the workers and lead are trained as two separate PPO models in alternating
rounds (train_workers_first.py), but with the roles reversed relative to
the lead-first sibling: each round, the workers train first against the
CURRENT frozen lead correction, then the lead trains against the
newly-updated frozen worker policy, and repeat.

Two environment classes share one underlying simulation:
- WorkerEnv (PettingZoo ParallelEnv): the workers' view, one agent per
  zone, no directive input (workers act independently and first). Each
  worker's reward is computed from its OWN provisional (pre-correction)
  proposal, evaluated as a pure counterfactual from the pre-step fleet --
  the same credit assignment the flat multi-agent sibling uses -- so
  workers are directly responsible for demand-matching and elder-care
  service on their own, rather than coasting on whatever the lead fixes
  afterward. The actual simulated fleet transition still uses the final,
  lead-corrected allocation.
- LeadEnv (gym.Env): the lead's view. Observes the workers' provisional
  allocation (computed by calling the frozen worker model) plus city-wide
  context, and outputs a bounded corrective adjustment added to the
  provisional logits before the final softmax. With a worker model
  attached, stepping LeadEnv alone runs the full combined (workers +
  correction) policy -- this is what evaluate/export scripts use.
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


# Bound on the lead's corrective adjustment, applied additively to the
# workers' provisional logits before the final softmax. Keeps the lead a
# "correction" rather than a full override of the workers' joint decision.
LEAD_CORRECTION_SCALE = 2.0


class _SimCore:
    """Shared simulation state/physics -- identical to the lead-first
    hierarchical sibling project (same reward, same effective-capacity and
    elder-care math), so the two architectures differ only in who acts
    first and how coordination is structured."""

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
        durations = np.array([self.dataset.avg_trip_duration(uid, t) for uid in self.dataset.unit_ids])
        return 60.0 / durations

    def _reset_core(self, rng: np.random.Generator):
        self.t = int(rng.integers(0, self.max_start))
        self.step_idx = 0
        self.fleet = np.full(self.n, self.total_vehicles / self.n, dtype=np.float64)

    def _evaluate_logits(self, logits: np.ndarray, forecast: np.ndarray, fleet: np.ndarray) -> dict:
        """Pure (non-mutating) outcome of applying `logits` from a given
        starting `fleet`, at the CURRENT simulated tick (self.t), without
        advancing any simulation state. Lets a caller score a hypothetical
        allocation -- e.g. a worker's own pre-correction proposal -- without
        committing it as the tick that actually happens."""
        target_fractions = _softmax(logits)
        target_fleet = target_fractions * self.total_vehicles
        trip_rate = self._trip_rate_all(self.t)

        required_demand = np.maximum(forecast * self.min_service_level, 0.0)
        required = required_demand / trip_rate
        required = np.where(self.elder_care_mask, np.maximum(required, self.min_vehicles), 0.0)
        shortfall = np.maximum(required - target_fleet, 0.0)
        relative_shortfall = np.divide(shortfall, required, out=np.zeros_like(shortfall), where=required > 0)
        elder_care_penalty = relative_shortfall * self.penalty_weight

        moved = np.abs(target_fleet - fleet)
        reposition_cost = moved * self.reposition_cost_per_vehicle

        actual_demand = self.demand_matrix[self.t]
        effective_capacity = target_fleet * trip_rate
        served = np.minimum(effective_capacity, actual_demand)
        fares = np.array([self.dataset.avg_fare(uid, self.t) for uid in self.dataset.unit_ids])
        revenue = served * fares * self.revenue_multiplier

        profit_per_zone = revenue - reposition_cost - elder_care_penalty

        return {
            "target_fleet": target_fleet,
            "actual_demand": actual_demand,
            "served": served,
            "revenue": revenue,
            "reposition_cost": reposition_cost,
            "elder_care_penalty": elder_care_penalty,
            "elder_care_violation": shortfall > 1e-6,
            "profit_per_zone": profit_per_zone,
        }

    def _advance(self, final_logits: np.ndarray, forecast: np.ndarray) -> dict:
        result = self._evaluate_logits(final_logits, forecast, self.fleet)
        self.fleet = result["target_fleet"]

        self.t += 1
        self.step_idx += 1
        truncated = self.step_idx >= self.episode_length

        return {**result, "forecast": forecast, "truncated": truncated}

    def _city_context(self, forecast: np.ndarray) -> tuple[float, float, float, float]:
        forecast_scale = max(forecast.max(), 1.0)
        hour = self.t % 24
        hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        city_fleet_pressure = self.fleet.sum() / self.total_vehicles
        city_demand_norm = forecast.sum() / (self.n * forecast_scale)
        return forecast_scale, city_fleet_pressure, city_demand_norm, (hour_sin, hour_cos)


class WorkerEnv(ParallelEnv, _SimCore):
    """The workers' view: one agent per zone, acting FIRST and
    independently (no directive input). Reward is computed from each
    worker's OWN provisional proposal (a pure counterfactual from the
    pre-step fleet), so a worker is scored on what it itself would have
    achieved -- not on whatever the lead's correction turns that into."""

    metadata = {"name": "av_fleet_allocation_workers_first_v0"}

    def __init__(self, dataset: DemandDataset, forecaster: BaseForecaster, cfg: dict, lead_model_path: str | None = None):
        super().__init__()
        self.render_mode = None
        self._init_core(dataset, forecaster, cfg)
        # A path, not a live model object: SuperSuit cloudpickles this whole
        # env, and SB3 model objects contain unpicklable internals (e.g.
        # torch locks). Lazily loaded from disk on first use instead.
        self.lead_model_path = lead_model_path
        self._lead_model = None

        self.possible_agents = [f"zone_{uid}" for uid in dataset.unit_ids]
        self.agents = list(self.possible_agents)
        self._idx_by_agent = {a: i for i, a in enumerate(self.possible_agents)}
        self._uid_by_agent = dict(zip(self.possible_agents, dataset.unit_ids))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # [own_fleet_frac, own_forecast_norm, own_elder_care_flag,
        #  own_trip_rate_norm, city_fleet_pressure, city_demand_norm, hour_sin, hour_cos]
        return spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

    def _get_lead_model(self):
        if self._lead_model is None and self.lead_model_path is not None:
            from stable_baselines3 import PPO
            self._lead_model = PPO.load(self.lead_model_path)
        return self._lead_model

    def _lead_correction(self, provisional_logits: np.ndarray, forecast: np.ndarray) -> np.ndarray:
        """Query the frozen lead for its correction to the workers'
        provisional joint action. Returns zeros (identity pass-through)
        when no lead model is attached yet (Stage A: workers-only warm-up)."""
        lead_model = self._get_lead_model()
        if lead_model is None:
            return np.zeros(self.n, dtype=np.float64)
        lead_obs = _lead_observation(self, provisional_logits, forecast)
        action, _ = lead_model.predict(lead_obs[None, :], deterministic=True)
        return np.tanh(action[0]).astype(np.float64) * LEAD_CORRECTION_SCALE

    def _observations(self, forecast: np.ndarray) -> dict:
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
        obs = self._observations(forecast)
        return obs, {agent: {} for agent in self.agents}

    def step(self, actions: dict):
        forecast = self._forecast_all(self.t)
        provisional_logits = np.array([actions[a][0] for a in self.agents], dtype=np.float64)
        correction = self._lead_correction(provisional_logits, forecast)
        final_logits = provisional_logits + correction

        # Workers are scored on their OWN proposal's outcome (a pure,
        # non-mutating evaluation from the pre-step fleet), not on what the
        # lead does to it afterward -- the same credit assignment the flat
        # multi-agent sibling uses, so each worker is directly responsible
        # for demand-matching and elder-care service on its own, with the
        # lead's correction learned as a genuine improvement on top rather
        # than something workers can coast on. The deployed/actual fleet
        # transition still uses the lead-corrected final_logits below.
        provisional_result = self._evaluate_logits(provisional_logits, forecast, self.fleet)
        result = self._advance(final_logits, forecast)

        rewards, terminations, truncations, infos = {}, {}, {}, {}
        for agent in self.agents:
            i = self._idx_by_agent[agent]
            rewards[agent] = float(provisional_result["profit_per_zone"][i]) / 1000.0
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
        obs = self._observations(next_forecast)
        return obs, rewards, terminations, truncations, infos


def _lead_observation(core: _SimCore, provisional_logits: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Shared by LeadEnv and WorkerEnv._lead_correction so both sides agree
    on exactly what the lead sees: the workers' provisional allocation
    (post joint-softmax, i.e. the fleet split the lead is correcting) plus
    the same city-wide context used elsewhere in this portfolio."""
    forecast_scale, _, _, (hour_sin, hour_cos) = core._city_context(forecast)
    trip_rate = core._trip_rate_all(core.t)
    provisional_fractions = _softmax(provisional_logits)
    obs = np.concatenate([
        provisional_fractions,  # provisional (pre-correction) allocation, already a 7-simplex
        core.fleet / core.total_vehicles,
        forecast / forecast_scale,
        core.elder_care_mask.astype(np.float64),
        trip_rate / max(trip_rate.max(), 1.0),
        [hour_sin, hour_cos],
    ])
    return obs.astype(np.float32)


class LeadEnv(gym.Env, _SimCore):
    """The lead/coordinator's view: observes the workers' provisional,
    already-combined allocation plus city-wide context, and outputs a
    bounded per-zone corrective adjustment SECOND. With a worker model
    attached, stepping LeadEnv alone runs the full combined (workers +
    correction) policy -- this is also what evaluate/export scripts use."""

    def __init__(self, dataset: DemandDataset, forecaster: BaseForecaster, cfg: dict, worker_model=None):
        super().__init__()
        self._init_core(dataset, forecaster, cfg)
        self.worker_model = worker_model

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n,), dtype=np.float32)
        # [provisional_fraction, fleet_frac, forecast_norm, elder_care, trip_rate_norm] per zone + hour_sin/cos
        obs_dim = 5 * self.n + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _worker_provisional(self, forecast: np.ndarray) -> np.ndarray:
        """Query the frozen worker model for each zone's raw action score,
        producing the provisional (pre-correction) joint logits. Returns
        zeros (i.e. a uniform provisional allocation) when no worker model
        is attached yet."""
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
                trip_rate[i] / trip_rate_scale,
                city_fleet_pressure,
                city_demand_norm,
                hour_sin,
                hour_cos,
            ], dtype=np.float32)
            for i in range(self.n)
        ])
        scores, _ = self.worker_model.predict(worker_obs, deterministic=True)
        return scores.reshape(self.n).astype(np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_core(self.np_random)
        forecast = self._forecast_all(self.t)
        provisional_logits = self._worker_provisional(forecast)
        return _lead_observation(self, provisional_logits, forecast), {}

    def step(self, lead_action: np.ndarray):
        lead_action = np.asarray(lead_action, dtype=np.float64)
        forecast = self._forecast_all(self.t)
        provisional_logits = self._worker_provisional(forecast)
        correction = np.tanh(lead_action) * LEAD_CORRECTION_SCALE
        final_logits = provisional_logits + correction

        result = self._advance(final_logits, forecast)
        profit = float(result["profit_per_zone"].sum())
        reward = profit / 1000.0

        next_forecast = self._forecast_all(self.t) if not result["truncated"] else forecast
        next_provisional = self._worker_provisional(next_forecast) if not result["truncated"] else provisional_logits
        obs = _lead_observation(self, next_provisional, next_forecast)
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
