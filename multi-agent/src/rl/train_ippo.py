"""Train a shared-parameter PPO policy across all zone-agents (independent
PPO / IPPO), using the standard PettingZoo -> SuperSuit -> stable-baselines3
parameter-sharing recipe: each zone-agent is treated as its own vectorized
sub-environment, and one PPO policy is trained across all of them at once,
exactly the "shared policy across homogeneous agents" pattern from Lin et
al. (2018) and validated as a strong simple MARL baseline by de Witt et al.
(2020, "Is Independent Learning All You Need in the StarCraft Multi-Agent
Challenge?").
"""

import argparse

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from src.forecasting.train import train_forecaster
from src.simulation.ma_fleet_env import AVFleetAllocationMAEnv
from src.utils.config import load_config, resolve_path


def make_vec_env(cfg: dict, dataset, forecaster):
    env = AVFleetAllocationMAEnv(dataset, forecaster, cfg)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # num_vec_envs=1: one copy of the (already N-agent) env is enough since
    # each of the N zone-agents is already its own parallel sub-environment.
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    return VecMonitor(env)


def train_ippo(cfg: dict, resume: bool = False):
    forecaster, train_ds, _ = train_forecaster(cfg)
    env = make_vec_env(cfg, train_ds, forecaster)

    rl_cfg = cfg["rl"]
    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    model_path = models_dir / "ippo_fleet_allocation.zip"
    tb_log = str(resolve_path(cfg, cfg["paths"]["outputs_dir"]) / "tb_logs")

    if resume and model_path.exists():
        print(f"[resume] continuing training from {model_path}")
        model = PPO.load(str(model_path), env=env, tensorboard_log=tb_log)
        model.learn(total_timesteps=rl_cfg["total_timesteps"], reset_num_timesteps=False)
    else:
        # seed intentionally omitted: SuperSuit's ConcatVecEnv doesn't implement
        # .seed(), which SB3's PPO calls internally whenever a seed is given.
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tb_log)
        model.learn(total_timesteps=rl_cfg["total_timesteps"])

    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(models_dir / "ippo_fleet_allocation"))
    print(f"[saved] IPPO shared policy -> {model_path}")

    return model, forecaster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_ippo(load_config(), resume=args.resume)
