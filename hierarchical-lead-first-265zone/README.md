# Robotaxi / AV Fleet Allocation -- Hierarchical, Lead-First (Lead + 260 Workers, 265-Zone)

Same objective as the sibling projects ([single-agent](../single-agent);
[multi-agent-265zone](../multi-agent-265zone), flat 261-agent): **maximize
profit, prioritize elder-care service**. Same hierarchical architecture as
[hierarchical-lead-first](../hierarchical-lead-first), scaled from the
7-borough validation run up to the full 261 active NYC TLC taxi zones: one
**lead/manager agent** sets a per-zone directive each hour, and **260
worker agents** (one per active zone, sharing one policy) each refine their
own zone's directive before the allocation is finalized.

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
- **Lead** observes the whole city and outputs one directive score per
  zone (261 of them). Its reward is the *same global profit/elder-care
  objective as the single-agent project* -- the lead literally optimizes
  the system-level target, with execution delegated to workers.
- **Workers** each observe their own zone's state plus the lead's
  directive for that zone, and output a scalar refinement. Final per-zone
  logit = directive + refinement, then the same joint softmax as the other
  projects reconciles everything into a feasible fleet allocation. Each
  worker's reward is its own zone's actual local profit outcome (not
  goal-adherence).

**Training** ([`src/rl/train_hierarchical.py`](src/rl/train_hierarchical.py)):
stable-baselines3 only trains one homogeneous policy at a time, so the lead
and workers can't be backpropagated through jointly. Instead they're
trained as two separate PPO models in **alternating rounds**: each round,
the lead trains against the current frozen worker policy, then the workers
train against the newly-updated frozen lead policy. `config.yaml`'s
`rl.rounds` controls how many alternations; the lead-first ordering trains
the lead first within each round, against the workers' most recently frozen
checkpoint.

## What's reused

Data and forecasting are identical to
[multi-agent-265zone](../multi-agent-265zone), at the full taxi-zone
resolution, for direct comparability. Raw data copied locally, not
re-downloaded.

## Setup and run

```bash
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.rl.train_hierarchical       # alternating lead<->worker training
python -m src.rl.evaluate_hierarchical    # vs. uniform / demand-proportional baselines
python -m src.rl.export_hierarchical_schedule
```
