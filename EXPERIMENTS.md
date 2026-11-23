# Experiment Manifest

Maps every architecture/zone-count combination reported in the paper
("Optimal Allocation Architectures for Robotaxi Fleet Management via
Multi-Agent Reinforcement Learning") to the exact folder, config
settings, and command that reproduces it. No code is duplicated here --
this is a pointer into the 9 folders already in this repo.

All twelve rows share the same equal-total-compute budget: **1,000,000
environment-interaction samples total** per architecture, however many
learners it uses.

| # | Paper setting | Folder | `config/config.yaml` settings | Run |
|---|---|---|---|---|
| 1 | Single-agent, 7 zones (borough) | [`single-agent/`](single-agent/) | `city.zone_level: borough` | `python scripts/run_pipeline.py` |
| 2 | Single-agent, 261 zones (taxi zone) | [`single-agent/`](single-agent/) | `city.zone_level: taxi_zone` (committed default) | `python scripts/run_pipeline.py` |
| 3 | Single-agent, 15 zones (KMeans) | [`single-agent-zone-sweep/`](single-agent-zone-sweep/) | `city.zone_level: kmeans`, `city.num_zones: 15` | `python scripts/run_pipeline.py` |
| 4 | Single-agent, 30 zones (KMeans) | [`single-agent-zone-sweep/`](single-agent-zone-sweep/) | `city.zone_level: kmeans`, `city.num_zones: 30` | `python scripts/run_pipeline.py` |
| 5 | Single-agent, 50 zones (KMeans) | [`single-agent-zone-sweep/`](single-agent-zone-sweep/) | `city.zone_level: kmeans`, `city.num_zones: 50` (committed default) | `python scripts/run_pipeline.py` |
| 6 | Flat multi-agent (IPPO), 7 zones | [`multi-agent/`](multi-agent/) | `city.zone_level: borough` (committed default); **set `rl.total_timesteps: 1000000`** (see note below) | `python -m src.rl.train_ippo` |
| 7 | Flat multi-agent (IPPO), 10 zones | [`multi-agent-10zone/`](multi-agent-10zone/) | `city.zone_level: kmeans`, `city.num_zones: 10` (committed default) | `python -m src.rl.train_ippo` |
| 8 | Flat multi-agent (IPPO), 261 zones | [`multi-agent-265zone/`](multi-agent-265zone/) | `city.zone_level: taxi_zone` (committed default) | `python -m src.rl.train_ippo` |
| 9 | Hierarchical lead-first, 7 zones | [`hierarchical-lead-first/`](hierarchical-lead-first/) | `city.zone_level: borough` (committed default) | `python -m src.rl.train_hierarchical` |
| 10 | Hierarchical lead-first, 261 zones | [`hierarchical-lead-first-265zone/`](hierarchical-lead-first-265zone/) | `city.zone_level: taxi_zone` (committed default) | `python -m src.rl.train_hierarchical` |
| 11 | Hierarchical workers-first, 7 zones | [`hierarchical-workers-first/`](hierarchical-workers-first/) | `city.zone_level: borough` (committed default) | `python -m src.rl.train_hierarchical` (or `python run_train.py`) |
| 12 | Hierarchical workers-first, 261 zones | [`hierarchical-workers-first-265zone/`](hierarchical-workers-first-265zone/) | `city.zone_level: taxi_zone` (committed default) | `python -m src.rl.train_hierarchical` (or `python run_train.py`) |

All four hierarchical rows (9-12) additionally use `rl.rounds: 3`,
`rl.lead_timesteps_per_round: 166667`, `rl.worker_timesteps_per_round:
166667` (committed default in all four folders) -- 3 alternating
best-response rounds, splitting the lead's 500,000-sample budget and the
workers' 500,000-sample budget into ~166,667-timestep installments each.

## Note on row 6 (`multi-agent/`)

This folder's committed `config.yaml` has `rl.total_timesteps: 2000000`,
left over from an earlier (non-equal-compute) experiment run before this
project's methodology was standardized. To reproduce the paper's
equal-total-compute number for the 7-zone flat multi-agent architecture,
override this to `1000000` before training -- matching rows 7 and 8,
which already have the correct committed value.

## Results

Aggregated comparison tables and charts (not per-run training logs) are in
[`results/`](results/): [`265zone_architecture_comparison.csv`](results/265zone_architecture_comparison.csv)
(rows 2, 8, 10, 12) and [`zone_count_sweep.csv`](results/zone_count_sweep.csv)
(rows 3-5).

## Reproducing from scratch

Run [`single-agent/`](single-agent/) first -- every other folder's
`src/data/preprocess.py` expects the same raw NYC TLC data, copied
locally rather than re-downloaded per folder (see each folder's `.gitignore`
and README). Each folder's own README has its full setup/run instructions.
