# Robotaxi / AV Fleet Allocation -- Multi-Agent Variant (10-Zone KMeans)

Same objective as [single-agent](../single-agent): **maximize profit, subject
to prioritizing elder-care service**. What changes here is the RL
architecture: instead of one monolithic policy choosing an allocation across
all zones at once, **each zone is its own agent**, sharing one policy network
(parameter sharing / "independent PPO", IPPO).

## Why multi-agent, and what design it follows

The single-agent project's action space grows with the number of zones --
one network has to output an allocation for every zone simultaneously. The
multi-agent framing decomposes that into N small per-zone decisions made by
a *shared* policy, which is the standard approach in the fleet-management RL
literature for handling large, homogeneous agent populations:

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

## Current scope: 10 geographic (KMeans) zones

This variant sits between the 7-borough and 261-taxi-zone granularities: the
261 official NYC TLC taxi zones are grouped into 10 geographic clusters via
KMeans on zone centroids
([`src/data/preprocess.py`](src/data/preprocess.py)), giving 10 homogeneous
agents -- finer-grained coordination than the 7-borough variant, without the
full cost of 261 parallel sub-environments under SuperSuit.

## Setup and run

```bash
pip install -r requirements.txt
python -m src.data.preprocess          # builds data/processed/demand.parquet (10 kmeans clusters)
python -m src.rl.train_ippo            # trains the shared PPO policy
python -m src.rl.evaluate_ippo         # compares vs. uniform / demand-proportional baselines
python -m src.rl.export_ippo_schedule  # per-zone-per-hour allocation CSV
```

`config.yaml`'s `rl.total_timesteps` is counted per-agent-step under
SuperSuit's vectorization (standard SB3 convention), not per environment
tick -- with 10 agents, the per-agent sample budget is divided across all 10.
