"""Alternating worker<->lead training (workers-first architecture).

Each round: train the workers (shared-parameter PPO over the 7 zone-agents,
via PettingZoo+SuperSuit) FIRST against the current frozen lead correction,
then train the lead (single-agent PPO) SECOND against the newly-updated
frozen worker policy. Repeat for `rl.rounds` rounds. This is the reverse
training order of the lead-first hierarchical sibling project, matching
the reversed roles in hierarchical_env.py: workers act first each tick,
the lead corrects their joint output second.
"""

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from src.forecasting.train import train_forecaster
from src.simulation.hierarchical_env import LeadEnv, WorkerEnv
from src.utils.config import load_config, resolve_path


def make_worker_vec_env(dataset, forecaster, cfg, lead_model_path):
    env = WorkerEnv(dataset, forecaster, cfg, lead_model_path=lead_model_path)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    return VecMonitor(env)


def train_hierarchical(cfg: dict):
    forecaster, train_ds, _ = train_forecaster(cfg)
    rl_cfg = cfg["rl"]
    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    tb_log = str(resolve_path(cfg, cfg["paths"]["outputs_dir"]) / "tb_logs")

    lead_model = None
    worker_model = None

    for round_idx in range(rl_cfg["rounds"]):
        print(f"\n===== Round {round_idx + 1}/{rl_cfg['rounds']}: training workers =====")
        lead_policy_path = str(models_dir / "lead_policy")
        worker_env = make_worker_vec_env(
            train_ds, forecaster, cfg,
            lead_model_path=(lead_policy_path + ".zip") if lead_model is not None else None,
        )
        if worker_model is None:
            worker_model = PPO("MlpPolicy", worker_env, verbose=1, tensorboard_log=tb_log)
        else:
            worker_model.set_env(worker_env)
        worker_model.learn(total_timesteps=rl_cfg["worker_timesteps_per_round"], reset_num_timesteps=False)
        worker_policy_path = str(models_dir / "worker_policy")
        worker_model.save(worker_policy_path)

        print(f"\n===== Round {round_idx + 1}/{rl_cfg['rounds']}: training lead =====")
        lead_env = Monitor(LeadEnv(train_ds, forecaster, cfg, worker_model=worker_model))
        if lead_model is None:
            lead_model = PPO("MlpPolicy", lead_env, verbose=1, tensorboard_log=tb_log)
        else:
            lead_model.set_env(lead_env)
        lead_model.learn(total_timesteps=rl_cfg["lead_timesteps_per_round"], reset_num_timesteps=False)
        lead_model.save(lead_policy_path)

    print(f"\n[saved] lead policy   -> {models_dir / 'lead_policy.zip'}")
    print(f"[saved] worker policy -> {models_dir / 'worker_policy.zip'}")
    return lead_model, worker_model, forecaster


if __name__ == "__main__":
    train_hierarchical(load_config())
