"""Train a PPO agent (stable-baselines3) on the AV fleet allocation environment."""

import argparse

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.forecasting.dataset import DemandDataset
from src.forecasting.train import train_forecaster
from src.simulation.fleet_env import AVFleetAllocationEnv
from src.utils.config import load_config, resolve_path


def make_env(cfg: dict, dataset: DemandDataset, forecaster):
    def _init():
        env = AVFleetAllocationEnv(dataset, forecaster, cfg)
        return Monitor(env)
    return _init


def train_agent(cfg: dict, resume: bool = False):
    forecaster, train_ds, _ = train_forecaster(cfg)

    env_fn = make_env(cfg, train_ds, forecaster)
    env = env_fn()

    rl_cfg = cfg["rl"]
    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    model_path = models_dir / "ppo_fleet_allocation.zip"
    tb_log = str(resolve_path(cfg, cfg["paths"]["outputs_dir"]) / "tb_logs")

    if resume and model_path.exists():
        print(f"[resume] continuing training from {model_path}")
        model = PPO.load(str(model_path), env=env, tensorboard_log=tb_log)
        model.learn(total_timesteps=rl_cfg["total_timesteps"], reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=rl_cfg["seed"], tensorboard_log=tb_log)
        model.learn(total_timesteps=rl_cfg["total_timesteps"])

    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(models_dir / "ppo_fleet_allocation"))
    print(f"[saved] PPO agent -> {model_path}")

    return model, forecaster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="continue training the existing saved model")
    args = parser.parse_args()
    train_agent(load_config(), resume=args.resume)
