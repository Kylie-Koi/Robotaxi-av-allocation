# Robotaxi / AV Fleet Allocation -- Multi-Agent Variant

Same objective as [robotaxi_av_allocation](../robotaxi_av_allocation): **maximize
profit, subject to prioritizing elder-care service**. What changes here is the
RL architecture: instead of one monolithic policy choosing an allocation
across all zones at once, **each zone is its own agent**, sharing one policy
network (parameter sharing / "independent PPO", IPPO).

## Why multi-agent, and what design it follows

The single-agent project's action space grows with the number of zones (7 at
borough level, 261 at taxi-zone level) -- one network has to output an
allocation for every zone simultaneously. The multi-agent framing decomposes
that into N small per-zone decisions made by a *shared* policy, which is the
standard approach in the fleet-management RL literature for handling
large, homogeneous agent populations:

- **Lin et al., KDD 2018** -- ["Efficient Large-Scale Fleet Management via
  Multi-Agent Deep RL"](https://arxiv.org/abs/1802.06444) treats each grid
  cell as an agent, shares one policy/value network across all grids, and
  appends a city-wide context vector to each agent's local state so agents
  implicitly coordinate without a full joint action space. Our
  [`ma_fleet_env.py`](src/simulation/ma_fleet_env.py) follows this pattern:
  zone = agent, shared network, local obs + global context.
- **de Witt et al., 2020** -- ["Is Independent Learning All You Need in the
  StarCraft Multi-Agent Challenge?"](https://arxiv.org/abs/2011.09533) shows
  independent/parameter-shared PPO (IPPO) is a strong, simple MARL baseline,
  which is why this project uses PettingZoo + SuperSuit's standard
  PettingZoo-to-stable-baselines3 recipe rather than a heavier framework.

See also, for broader context gathered while designing this: [Robo-taxi Fleet
Coordination at Scale via RL](https://arxiv.org/abs/2504.06125), [constrained
mean-field RL for ride-sourcing rebalancing with service-accessibility
guarantees](https://www.sciencedirect.com/science/article/pii/S0968090X26001932)
(paywalled; abstract-level only), and [MADDPG for ride-hailing
rebalancing](https://arxiv.org/abs/2412.00276).

## Reused vs. new

Reused unchanged from the single-agent project (same data, same forecasting):
`src/data/`, `src/forecasting/`, `src/utils/config.py`. Data files were
copied locally rather than re-downloaded.

New for the multi-agent architecture:
- [`src/simulation/ma_fleet_env.py`](src/simulation/ma_fleet_env.py) -- a
  PettingZoo `ParallelEnv` where each zone is an agent. Per-agent reward is
  that zone's own (revenue − reposition cost − elder-care penalty), so
  credit assignment is local, but joint feasibility (can't allocate more
  than `total_vehicles`) is still enforced centrally via one softmax over
  all agents' actions each tick -- same mechanism as the single-agent env,
  just applied across agent outputs instead of one policy's output vector.
- [`src/rl/train_ippo.py`](src/rl/train_ippo.py) -- wraps the PettingZoo env
  with SuperSuit (`pettingzoo_env_to_vec_env_v1` + `concat_vec_envs_v1`) so
  stable-baselines3's PPO trains one shared policy across all zone-agents.
- [`src/rl/evaluate_ippo.py`](src/rl/evaluate_ippo.py),
  [`export_ippo_schedule.py`](src/rl/export_ippo_schedule.py) -- same
  baseline comparisons and schedule export as the single-agent project.

## Current scope: borough-level (7 agents), not yet the 261 taxi zones

The single-agent project itself was built up incrementally (borough level
first, to validate the RL loop, then taxi-zone level once that worked). This
multi-agent variant follows the same path: starting at borough granularity
to validate the multi-agent architecture (parameter sharing, per-zone credit
assignment, joint-action feasibility) before scaling to 261 zones, which
would mean 261 parallel sub-environments per step under SuperSuit.

## Setup and run

```bash
pip install -r requirements.txt
python -m src.data.preprocess          # builds data/processed/demand.parquet (borough level)
python -m src.rl.train_ippo            # trains the shared PPO policy
python -m src.rl.evaluate_ippo         # compares vs. uniform / demand-proportional baselines
python -m src.rl.export_ippo_schedule  # per-zone-per-hour allocation CSV
```

`config.yaml`'s `rl.total_timesteps` is counted per-agent-step under
SuperSuit's vectorization (standard SB3 convention), not per environment
tick -- with 7 agents, 200,000 timesteps is roughly 28,500 real simulated
hours, not 200,000.
