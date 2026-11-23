# Robotaxi / AV Fleet Allocation

## Quick start

### Single-agent (one PPO policy for all zones)

cd single-agent
pip install -r requirements.txt
python scripts/run_pipeline.py

## NOTE: RUN single agent first bc multi-agent data files are copied locally, not redownloaded :)

### Multi-agent (one agent per zone, shared policy, IPPO)

cd multi-agent
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.rl.train_ippo
python -m src.rl.evaluate_ippo
python -m src.rl.export_ippo_schedule

### -------------------------------------------------------------------

Forecasts robotaxi demand in a city and optimizes fleet allocation with
reinforcement learning, maximizing profit subject to a service-level
constraint that prioritizes elder-care locations (nursing homes, assisted
living facilities).

This repo contains two variants of the project:

- **[`single-agent/`](single-agent/)** — one policy (PPO) chooses the fleet
  allocation across all zones at once.
- **[`multi-agent/`](multi-agent/)** — the same objective, but each zone is
  its own agent sharing one policy network (independent PPO / IPPO). See
  [`multi-agent/README.md`](multi-agent/README.md) for why this variant
  exists and the papers it follows (Lit review inspiration).

Each subfolder is self-contained: its own `README.md`, `requirements.txt`,
and `config/config.yaml`. See the subfolder READMEs for setup and run
instructions.
