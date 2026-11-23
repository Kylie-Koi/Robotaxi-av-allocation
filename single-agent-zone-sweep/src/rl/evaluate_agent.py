"""Evaluate a trained PPO agent on held-out data, compared against two simple
baselines: uniform allocation and demand-proportional allocation.

Evaluated over 10 fixed seeds (reproducible test weeks) instead of one
unseeded episode, reporting mean +/- std so comparisons across architectures
aren't just noise from which random week got sampled.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.forecasting.dataset import DemandDataset
from src.forecasting.train import train_forecaster
from src.simulation.fleet_env import AVFleetAllocationEnv, _softmax
from src.utils.config import load_config, resolve_path

EVAL_SEEDS = list(range(10))


def run_episode(env: AVFleetAllocationEnv, policy_fn, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    totals = {"profit": 0.0, "revenue": 0.0, "reposition_cost": 0.0,
              "elder_care_penalty": 0.0, "elder_care_violations": 0,
              "served_demand": 0.0, "actual_demand": 0.0}
    done = False
    while not done:
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        for k in totals:
            totals[k] += info.get(k, 0.0)
        done = terminated or truncated
    totals["served_demand_rate"] = (
        totals["served_demand"] / totals["actual_demand"] if totals["actual_demand"] > 0 else float("nan")
    )
    return totals


def run_multi_episode(env, policy_fn, seeds=EVAL_SEEDS) -> dict:
    episodes = [run_episode(env, policy_fn, seed) for seed in seeds]
    keys = episodes[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([ep[k] for ep in episodes], dtype=np.float64)
        summary[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return {"episodes": episodes, "summary": summary}


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="override path to the trained PPO model .zip")
    parser.add_argument("--out", default=None, help="override path to write the JSON results summary")
    args = parser.parse_args()

    cfg = load_config()
    forecaster, _, test_ds = train_forecaster(cfg)
    env = AVFleetAllocationEnv(test_ds, forecaster, cfg)

    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    model_path = Path(args.model_path) if args.model_path else models_dir / "ppo_fleet_allocation.zip"

    policies = {"uniform": uniform_policy, "demand_proportional": proportional_policy}
    if model_path.exists():
        model = PPO.load(str(model_path))
        policies["ppo"] = ppo_policy(model)
    else:
        print(f"[warn] no trained PPO model found at {model_path}; skipping")

    all_results = {}
    for name, policy_fn in policies.items():
        result = run_multi_episode(env, policy_fn)
        all_results[name] = result
        print(f"\n[{name}] (mean +/- std over {len(EVAL_SEEDS)} seeded episodes)")
        for k, v in result["summary"].items():
            print(f"  {k}: {v['mean']:.2f} +/- {v['std']:.2f}")

    out_path = Path(args.out) if args.out else resolve_path(cfg, cfg["paths"]["outputs_dir"]) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] eval results -> {out_path}")


if __name__ == "__main__":
    main()
