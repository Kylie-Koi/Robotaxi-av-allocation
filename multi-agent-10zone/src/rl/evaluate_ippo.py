"""Evaluate the shared IPPO policy on held-out data against uniform and
demand-proportional baselines -- same comparison as the single-agent
project's evaluate_agent.py, run through the multi-agent env instead.

Evaluated over 10 fixed seeds (reproducible test weeks) instead of one
unseeded episode, reporting mean +/- std.
"""

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.forecasting.train import train_forecaster
from src.simulation.ma_fleet_env import AVFleetAllocationMAEnv
from src.utils.config import load_config, resolve_path

# Observation layout from ma_fleet_env._observations():
# [own_fleet_frac, own_forecast_norm, own_elder_care_flag, own_trip_rate_norm, city_fleet_pressure, city_demand_norm, hour_sin, hour_cos]
OWN_FORECAST_IDX = 1

EVAL_SEEDS = list(range(10))


def run_episode(env: AVFleetAllocationMAEnv, policy_fn, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    totals = {"profit": 0.0, "revenue": 0.0, "reposition_cost": 0.0,
              "elder_care_penalty": 0.0, "elder_care_violations": 0,
              "served_demand": 0.0, "actual_demand": 0.0}
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
            totals["served_demand"] += info["served_demand"]
            totals["actual_demand"] += info["actual_demand"]
        done = len(agents) == 0 or all(truncations[a] for a in agents)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="override path to write the JSON results summary")
    args = parser.parse_args()

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
