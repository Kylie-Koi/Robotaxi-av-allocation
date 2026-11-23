"""Evaluate the combined lead+worker hierarchical policy against uniform and
demand-proportional baselines. Baselines bypass the hierarchy entirely (no
worker model attached, raw logits passed as the "lead action"), reproducing
the exact same baselines used in the single-agent and flat multi-agent
sibling projects.

Evaluated over 10 fixed seeds (reproducible test weeks) instead of one
unseeded episode, reporting mean +/- std.
"""

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.forecasting.train import train_forecaster
from src.simulation.hierarchical_env import LeadEnv
from src.utils.config import load_config, resolve_path

EVAL_SEEDS = list(range(10))


def run_episode(env: LeadEnv, policy_fn, seed: int) -> dict:
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
        done = truncated
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


def uniform_policy(obs, env):
    return np.zeros(env.n, dtype=np.float32)


def proportional_policy(obs, env):
    forecast = obs[env.n:2 * env.n]
    return forecast.astype(np.float32)


def hierarchical_policy(lead_model):
    def _fn(obs, env):
        action, _ = lead_model.predict(obs, deterministic=True)
        return action
    return _fn


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="override path to write the JSON results summary")
    args = parser.parse_args()

    cfg = load_config()
    forecaster, _, test_ds = train_forecaster(cfg)

    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    lead_path = models_dir / "lead_policy.zip"
    worker_path = models_dir / "worker_policy.zip"

    baseline_env = LeadEnv(test_ds, forecaster, cfg, worker_model=None)
    runs = [("uniform", baseline_env, uniform_policy), ("demand_proportional", baseline_env, proportional_policy)]

    if lead_path.exists() and worker_path.exists():
        lead_model = PPO.load(str(lead_path))
        worker_model = PPO.load(str(worker_path))
        combined_env = LeadEnv(test_ds, forecaster, cfg, worker_model=worker_model)
        runs.append(("hierarchical (lead+workers)", combined_env, hierarchical_policy(lead_model)))
    else:
        print(f"[warn] trained lead/worker models not found at {models_dir}; run train_hierarchical.py first")

    all_results = {}
    for name, env, policy_fn in runs:
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
