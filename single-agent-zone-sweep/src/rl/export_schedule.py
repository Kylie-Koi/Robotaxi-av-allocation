"""Run a policy for one episode and export the resulting zone x hour fleet
allocation schedule to CSV (plus a quick plot), so the RL policy's decisions
can actually be inspected rather than just summarized as episode totals.

Usage:
    python -m src.rl.export_schedule              # uses trained PPO if present
    python -m src.rl.export_schedule --policy proportional
"""

import argparse

import pandas as pd
from stable_baselines3 import PPO

from src.forecasting.train import train_forecaster
from src.rl.evaluate_agent import proportional_policy, uniform_policy
from src.simulation.fleet_env import AVFleetAllocationEnv
from src.utils.config import load_config, resolve_path


def _load_borough_map(cfg: dict) -> dict:
    """Map unit_id -> borough. At taxi_zone granularity unit_id is a
    LocationID, looked up against the zone table; at borough granularity
    unit_name already *is* the borough, so callers fall back to that."""
    raw_dir = resolve_path(cfg, cfg["data"]["raw_dir"])
    lookup = pd.read_csv(raw_dir / "taxi_zone_lookup.csv")
    return dict(zip(lookup["LocationID"], lookup["Borough"].fillna("Unknown")))


def run_and_record(env: AVFleetAllocationEnv, dataset, policy_fn, cfg: dict) -> pd.DataFrame:
    borough_by_id = _load_borough_map(cfg)
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        t_used = env.t  # timestamp this step's decision applies to
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        timestamp = dataset.timestamps[t_used]

        for i, uid in enumerate(dataset.unit_ids):
            rows.append({
                "timestamp": timestamp,
                "unit_id": uid,
                "unit_name": dataset.unit_names[uid],
                "borough": borough_by_id.get(uid, dataset.unit_names[uid]),
                "is_elder_care": bool(dataset.elder_care_flags[uid]),
                "target_fleet": float(info["zone_target_fleet"][i]),
                "forecast_demand": float(info["zone_forecast"][i]),
                "actual_demand": float(info["zone_actual_demand"][i]),
                "served_demand": float(info["zone_served"][i]),
            })
        done = terminated or truncated

    return pd.DataFrame(rows)


def plot_schedule(schedule: pd.DataFrame, out_path, max_zones_per_borough: int = 2):
    import matplotlib.pyplot as plt

    # Picking a global top-N by demand would be all-Manhattan (it dwarfs
    # every other borough's demand), so instead take the top zones *within
    # each* borough -- this keeps Queens/Brooklyn/Bronx/EWR/Staten Island
    # visible on the same chart instead of drowned out.
    avg_demand = schedule.groupby(["borough", "unit_id"])["actual_demand"].mean()
    top_units = (
        avg_demand.groupby("borough", group_keys=False)
        .apply(lambda s: s.sort_values(ascending=False).head(max_zones_per_borough))
        .index.get_level_values("unit_id")
    )
    top = schedule[schedule["unit_id"].isin(top_units)].copy()
    name_by_id = top.drop_duplicates("unit_id").set_index("unit_id")["unit_name"]
    borough_by_id = top.drop_duplicates("unit_id").set_index("unit_id")["borough"]
    top["label"] = top["unit_id"].map(lambda uid: f"{name_by_id[uid]} ({borough_by_id[uid]})")

    pivot = top.pivot(index="timestamp", columns="label", values="target_fleet")
    ax = pivot.plot(figsize=(13, 6), title=f"RL fleet allocation over time (top {max_zones_per_borough} zones per borough)")
    ax.set_ylabel("Vehicles allocated")
    ax.set_xlabel("Time")
    ax.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["ppo", "uniform", "proportional"], default="ppo")
    args = parser.parse_args()

    cfg = load_config()
    forecaster, _, test_ds = train_forecaster(cfg)
    env = AVFleetAllocationEnv(test_ds, forecaster, cfg)

    if args.policy == "ppo":
        model_path = resolve_path(cfg, cfg["paths"]["models_dir"]) / "ppo_fleet_allocation.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"No trained PPO model at {model_path}; run train_agent.py first "
                                     f"or pass --policy uniform/proportional")
        model = PPO.load(str(model_path))
        policy_fn = lambda obs, env: model.predict(obs, deterministic=True)[0]
    elif args.policy == "uniform":
        policy_fn = uniform_policy
    else:
        policy_fn = proportional_policy

    schedule = run_and_record(env, test_ds, policy_fn, cfg)

    # Sum across zones within each borough per hour first, then average over
    # time -- gives "total vehicles/demand in this borough", not "per zone".
    per_step = schedule.groupby(["timestamp", "borough"]).agg(
        total_target_fleet=("target_fleet", "sum"),
        total_forecast=("forecast_demand", "sum"),
        total_served=("served_demand", "sum"),
        total_actual=("actual_demand", "sum"),
    )
    num_zones = schedule.groupby("borough")["unit_id"].nunique()
    borough_summary = per_step.groupby("borough").mean().round(1)
    borough_summary["num_zones"] = num_zones
    print(borough_summary.to_string())

    outputs_dir = resolve_path(cfg, cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = outputs_dir / f"allocation_schedule_{args.policy}.csv"
    schedule.to_csv(csv_path, index=False)
    print(f"[saved] schedule -> {csv_path} ({len(schedule)} rows)")

    png_path = outputs_dir / f"allocation_schedule_{args.policy}.png"
    plot_schedule(schedule, png_path)
    print(f"[saved] plot -> {png_path}")


if __name__ == "__main__":
    main()
