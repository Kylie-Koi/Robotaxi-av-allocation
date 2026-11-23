"""Run the combined lead+worker policy for one episode and export the
resulting zone x hour allocation schedule to CSV.
"""

import pandas as pd
from stable_baselines3 import PPO

from src.forecasting.train import train_forecaster
from src.rl.evaluate_hierarchical import hierarchical_policy
from src.simulation.hierarchical_env import LeadEnv
from src.utils.config import load_config, resolve_path


def run_and_record(env: LeadEnv, dataset, policy_fn) -> pd.DataFrame:
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        t_used = env.t
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        timestamp = dataset.timestamps[t_used]
        for i, uid in enumerate(dataset.unit_ids):
            rows.append({
                "timestamp": timestamp,
                "unit_id": uid,
                "unit_name": dataset.unit_names[uid],
                "is_elder_care": bool(dataset.elder_care_flags[uid]),
                "target_fleet": float(info["zone_target_fleet"][i]),
                "forecast_demand": float(info["zone_forecast"][i]),
                "actual_demand": float(info["zone_actual_demand"][i]),
                "served_demand": float(info["zone_served"][i]),
            })
        done = truncated
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    forecaster, _, test_ds = train_forecaster(cfg)

    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    lead_model = PPO.load(str(models_dir / "lead_policy.zip"))
    worker_model = PPO.load(str(models_dir / "worker_policy.zip"))
    env = LeadEnv(test_ds, forecaster, cfg, worker_model=worker_model)

    schedule = run_and_record(env, test_ds, hierarchical_policy(lead_model))

    summary = schedule.groupby("unit_name").agg(
        avg_target_fleet=("target_fleet", "mean"),
        avg_forecast=("forecast_demand", "mean"),
        avg_served=("served_demand", "mean"),
        avg_actual=("actual_demand", "mean"),
    ).round(1)
    print(summary.to_string())

    outputs_dir = resolve_path(cfg, cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = outputs_dir / "hierarchical_allocation_schedule.csv"
    schedule.to_csv(csv_path, index=False)
    print(f"[saved] schedule -> {csv_path} ({len(schedule)} rows)")


if __name__ == "__main__":
    main()
