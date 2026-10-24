"""Evaluate a trained PPO agent on held-out data, compared against two simple
baselines: uniform allocation and demand-proportional allocation.
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.forecasting.dataset import DemandDataset
from src.forecasting.train import train_forecaster
from src.simulation.fleet_env import AVFleetAllocationEnv, _softmax
from src.utils.config import load_config, resolve_path


def run_episode(env: AVFleetAllocationEnv, policy_fn) -> dict:
    obs, _ = env.reset()
    totals = {"profit": 0.0, "revenue": 0.0, "reposition_cost": 0.0,
              "elder_care_penalty": 0.0, "elder_care_violations": 0}
    done = False
    while not done:
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        for k in totals:
            totals[k] += info[k]
        done = terminated or truncated
    return totals


def ppo_policy(model):
    def _fn(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return _fn


def uniform_policy(_obs, env):
    return np.zeros(env.n, dtype=np.float32)  # softmax(0) = uniform


def proportional_policy(obs, env):
    forecast = obs[env.n:2 * env.n]
    return forecast.astype(np.float32)  # softmax favors higher-forecast zones


def main():
    cfg = load_config()
    forecaster, _, test_ds = train_forecaster(cfg)
    env = AVFleetAllocationEnv(test_ds, forecaster, cfg)

    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    model_path = models_dir / "ppo_fleet_allocation.zip"

    policies = {"uniform": uniform_policy, "demand_proportional": proportional_policy}
    if model_path.exists():
        model = PPO.load(str(model_path))
        policies["ppo"] = ppo_policy(model)
    else:
        print(f"[warn] no trained PPO model found at {model_path}; skipping")

    for name, policy_fn in policies.items():
        results = run_episode(env, policy_fn)
        print(f"\n[{name}]")
        for k, v in results.items():
            print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
