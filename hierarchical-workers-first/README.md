# Robotaxi / AV Fleet Allocation -- Hierarchical, Workers-First (Lead + 7 Workers)

Same objective as the sibling projects
([single-agent](../single-agent); [multi-agent](../multi-agent), flat
7-agent): **maximize profit, prioritize elder-care service**. Same
hierarchical architecture as [hierarchical-lead-first](../hierarchical-lead-first)
-- one **lead/manager agent** plus **7 worker agents** (one per borough,
sharing one policy) -- but with the opposite execution order: **workers act
first**, independently, and the **lead applies a bounded corrective
adjustment second**, instead of the lead setting a directive that workers
refine.

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
- **Lead** observes the whole city (same 23-dim observation as the
  single-agent project) plus the workers' proposed allocation, and outputs a
  bounded per-zone corrective adjustment applied on top of it. Its reward is
  the same global profit/elder-care objective as the single-agent project.

**Training** ([`src/rl/train_hierarchical.py`](src/rl/train_hierarchical.py)):
same alternating-best-response scheme as the lead-first sibling -- stable-
baselines3 only trains one homogeneous policy at a time, so the lead and
workers are trained as two separate PPO models across `rl.rounds` alternating
rounds, each training against the other tier's most recently frozen
checkpoint. The only difference from
[hierarchical-lead-first](../hierarchical-lead-first) is which tier trains
(and acts) first within each round -- workers against the frozen lead before
the lead trains against the newly-updated workers, rather than the reverse --
isolating the effect of hierarchy direction on the final policy.

## What's reused

Data, forecasting, and the borough-level (7-zone), 4,500-vehicle, elder-care
config are identical to [multi-agent](../multi-agent), for direct
comparability. Raw data copied locally, not re-downloaded.

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
