# Robotaxi / AV Fleet Allocation -- Hierarchical, Workers-First (Lead + 260 Workers, 265-Zone)

Same objective as the sibling projects ([single-agent](../single-agent);
[multi-agent-265zone](../multi-agent-265zone), flat 261-agent): **maximize
profit, prioritize elder-care service**. Same hierarchical architecture as
[hierarchical-workers-first](../hierarchical-workers-first), scaled from the
7-borough validation run up to the full 261 active NYC TLC taxi zones: **260
worker agents** (one per active zone, sharing one policy) act first,
independently, and the **lead agent** applies a bounded corrective
adjustment second.

## Design

Follows **Ahilan & Dayan (2019), ["Feudal Multi-Agent Hierarchies for
Cooperative Reinforcement
Learning"](https://ala2019.vub.ac.be/papers/ALA2019_paper_5.pdf)** and
**Vezhnevets et al. (2017), ["FeUdal Networks for Hierarchical Reinforcement
Learning"](https://proceedings.mlr.press/v70/vezhnevets17a/vezhnevets17a.pdf)**,
same as the lead-first sibling, but isolates which tier moves first within
the hierarchy
([`src/simulation/hierarchical_env.py`](src/simulation/hierarchical_env.py)):
- **Workers** each observe their own zone's state and act independently
  first, with no directive from the lead yet available. Reward is each
  worker's own zone's actual local profit outcome.
- **Lead** observes the whole city plus the workers' proposed allocation
  across all 261 zones, and outputs a bounded per-zone corrective
  adjustment applied on top of it. Its reward is the same global
  profit/elder-care objective as the single-agent project.

**Training** ([`src/rl/train_hierarchical.py`](src/rl/train_hierarchical.py)):
same alternating-best-response scheme as the lead-first sibling -- the lead
and workers are trained as two separate PPO models across `rl.rounds`
alternating rounds, each training against the other tier's most recently
frozen checkpoint. The only difference from
[hierarchical-lead-first-265zone](../hierarchical-lead-first-265zone) is
which tier trains (and acts) first within each round -- workers against the
frozen lead before the lead trains against the newly-updated workers, rather
than the reverse -- isolating the effect of hierarchy direction at this
scale.

## What's reused

Data and forecasting are identical to
[multi-agent-265zone](../multi-agent-265zone), at the full taxi-zone
resolution, for direct comparability. Raw data copied locally, not
re-downloaded.

## Setup and run

```bash
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.rl.train_hierarchical       # alternating worker<->lead training
python -m src.rl.evaluate_hierarchical    # vs. uniform / demand-proportional baselines
python -m src.rl.export_hierarchical_schedule
```

`run_train.py` at the repo root is an equivalent entrypoint to
`python -m src.rl.train_hierarchical` for environments where module
execution from the package root is more convenient.
