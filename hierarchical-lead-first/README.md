# Robotaxi / AV Fleet Allocation -- Hierarchical (Lead + 7 Workers)

Same objective as the sibling projects ([single-agent](../single-agent);
[multi-agent](../multi-agent), flat 7-agent): **maximize profit, prioritize
elder-care service**. Here the
RL architecture is hierarchical: one **lead/manager agent** sets a per-zone
directive each hour, and **7 worker agents** (one per borough, sharing one
policy) each refine their own zone's directive before the allocation is
finalized.

## Design

Follows **Ahilan & Dayan (2019), ["Feudal Multi-Agent Hierarchies for
Cooperative Reinforcement
Learning"](https://ala2019.vub.ac.be/papers/ALA2019_paper_5.pdf)**: one
manager observing global state issues a goal/subgoal to each of several
simultaneously-acting workers, who act conditioned on their own goal
component. The underlying single-agent idea is from **Vezhnevets et al.
(2017), ["FeUdal Networks for Hierarchical Reinforcement
Learning"](https://proceedings.mlr.press/v70/vezhnevets17a/vezhnevets17a.pdf)**.

Adapted to this domain ([`src/simulation/hierarchical_env.py`](src/simulation/hierarchical_env.py)):
- **Lead** observes the whole city (same 23-dim observation as the
  single-agent project) and outputs one directive score per zone. Its
  reward is the *same global profit/elder-care objective as the
  single-agent project* -- the lead literally optimizes the system-level
  target, with execution delegated to workers.
- **Workers** each observe their own zone's state plus the lead's
  directive for that zone, and output a scalar refinement. Final per-zone
  logit = directive + refinement, then the same joint softmax as the other
  projects reconciles everything into a feasible fleet allocation. Each
  worker's reward is its own zone's actual local profit outcome (not
  goal-adherence), since in this domain the "right" allocation is defined
  by the profit/elder-care objective itself, not an abstract subgoal.

**Training** ([`src/rl/train_hierarchical.py`](src/rl/train_hierarchical.py)):
stable-baselines3 only trains one homogeneous policy at a time, so the lead
and workers can't be backpropagated through jointly. Instead they're
trained as two separate PPO models in **alternating rounds**: each round,
the lead trains against the current frozen worker policy, then the workers
train against the newly-updated frozen lead policy. Ahilan & Dayan's own
ablations found that co-located/joint manager-worker training beats
training them in isolation -- this alternation is the practical
approximation of that achievable with off-the-shelf single-agent PPO.
`config.yaml`'s `rl.rounds` controls how many alternations.

## What's reused

Data, forecasting, and the borough-level (7-zone), 4,500-vehicle, elder-care
config are identical to [multi-agent](../multi-agent), for direct
comparability. Raw data copied locally, not re-downloaded.

## Setup and run

```bash
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.rl.train_hierarchical       # alternating lead<->worker training
python -m src.rl.evaluate_hierarchical    # vs. uniform / demand-proportional baselines
python -m src.rl.export_hierarchical_schedule
```
