"""Run the shared IPPO policy for one episode and export the resulting
zone x hour allocation schedule to CSV + plot -- multi-agent counterpart of
the single-agent project's export_schedule.py.
"""

import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.forecasting.train import train_forecaster
from src.rl.evaluate_ippo import ippo_policy, proportional_policy, uniform_policy
from src.simulation.ma_fleet_env import AVFleetAllocationMAEnv
from src.utils.config import load_config, resolve_path


def run_and_record(env: AVFleetAllocationMAEnv, dataset, policy_fn) -> pd.DataFrame:
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        agents = list(env.agents)
        t_used = env.t
        actions = policy_fn(obs, agents)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        timestamp = dataset.timestamps[t_used]

        for agent in agents:
            uid = env._uid_by_agent[agent]
            info = infos[agent]
            rows.append({
                "timestamp": timestamp,
                "unit_id": uid,
                "unit_name": dataset.unit_names[uid],
                "is_elder_care": bool(dataset.elder_care_flags[uid]),
                "target_fleet": info["target_fleet"],
                "forecast_demand": info["forecast"],
                "actual_demand": info["actual_demand"],
                "served_demand": info["served_demand"],
            })
        done = len(agents) == 0 or all(truncations[a] for a in agents)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["ippo", "uniform", "proportional"], default="ippo")
    args = parser.parse_args()

    cfg = load_config()
    forecaster, _, test_ds = train_forecaster(cfg)
    env = AVFleetAllocationMAEnv(test_ds, forecaster, cfg)

    if args.policy == "ippo":
        model_path = resolve_path(cfg, cfg["paths"]["models_dir"]) / "ippo_fleet_allocation.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"No trained IPPO model at {model_path}; run train_ippo.py first")
        model = PPO.load(str(model_path))
        policy_fn = ippo_policy(model)
    elif args.policy == "uniform":
        policy_fn = uniform_policy
    else:
        policy_fn = proportional_policy

    schedule = run_and_record(env, test_ds, policy_fn)

    outputs_dir = resolve_path(cfg, cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summary = schedule.groupby("unit_name").agg(
        avg_target_fleet=("target_fleet", "mean"),
        avg_forecast=("forecast_demand", "mean"),
        avg_served=("served_demand", "mean"),
        avg_actual=("actual_demand", "mean"),
    ).round(1)
    print(summary.to_string())

    csv_path = outputs_dir / f"ma_allocation_schedule_{args.policy}.csv"
    schedule.to_csv(csv_path, index=False)
    print(f"[saved] schedule -> {csv_path} ({len(schedule)} rows)")


if __name__ == "__main__":
    main()
