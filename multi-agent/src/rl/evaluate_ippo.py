"""Evaluate the shared IPPO policy on held-out data against uniform and
demand-proportional baselines -- same comparison as the single-agent
project's evaluate_agent.py, run through the multi-agent env instead.
"""

import numpy as np
from stable_baselines3 import PPO

from src.forecasting.train import train_forecaster
from src.simulation.ma_fleet_env import AVFleetAllocationMAEnv
from src.utils.config import load_config, resolve_path

# Observation layout from ma_fleet_env._observations():
# [own_fleet_frac, own_forecast_norm, own_elder_care_flag, city_fleet_pressure, city_demand_norm, hour_sin, hour_cos]
OWN_FORECAST_IDX = 1


def run_episode(env: AVFleetAllocationMAEnv, policy_fn) -> dict:
    obs, _ = env.reset()
    totals = {"profit": 0.0, "revenue": 0.0, "reposition_cost": 0.0,
              "elder_care_penalty": 0.0, "elder_care_violations": 0}
    done = False
    while not done:
        agents = list(env.agents)
        actions = policy_fn(obs, agents)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        for agent in agents:
            info = infos[agent]
            totals["profit"] += info["profit"]
            totals["revenue"] += info["revenue"]
            totals["reposition_cost"] += info["reposition_cost"]
            totals["elder_care_penalty"] += info["elder_care_penalty"]
            totals["elder_care_violations"] += int(info["elder_care_violation"])
        done = len(agents) == 0 or all(truncations[a] for a in agents)
    return totals


def ippo_policy(model):
    def _fn(obs, agents):
        stacked = np.stack([obs[a] for a in agents])
        actions, _ = model.predict(stacked, deterministic=True)
        return {a: actions[i] for i, a in enumerate(agents)}
    return _fn


def uniform_policy(obs, agents):
    return {a: np.zeros(1, dtype=np.float32) for a in agents}


def proportional_policy(obs, agents):
    return {a: np.array([obs[a][OWN_FORECAST_IDX]], dtype=np.float32) for a in agents}


def main():
    cfg = load_config()
    forecaster, _, test_ds = train_forecaster(cfg)
    env = AVFleetAllocationMAEnv(test_ds, forecaster, cfg)

    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    model_path = models_dir / "ippo_fleet_allocation.zip"

    policies = {"uniform": uniform_policy, "demand_proportional": proportional_policy}
    if model_path.exists():
        model = PPO.load(str(model_path))
        policies["ippo"] = ippo_policy(model)
    else:
        print(f"[warn] no trained IPPO model at {model_path}; skipping")

    for name, policy_fn in policies.items():
        results = run_episode(env, policy_fn)
        print(f"\n[{name}]")
        for k, v in results.items():
            print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
