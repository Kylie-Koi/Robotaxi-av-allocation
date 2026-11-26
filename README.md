# Robotaxi / AV Fleet Allocation
## Keep scrolling for quickstart :O

Forecasts robotaxi demand in a city and optimizes fleet allocation with
reinforcement learning, maximizing profit subject to a service-level
constraint that prioritizes elder-care locations (nursing homes, assisted
living facilities). This repo treats the choice of multi-agent
organizational structure as an economic resource-allocation problem in its
own right: given a fixed computational training budget (1,000,000
environment-interaction samples), how should decision-making authority be
divided among agents, and how many zones should the city be decomposed
into?


## Architectures

| Family | Folder(s) | Zones | Learners |
|---|---|---|---|
| Single-agent | [`single-agent/`](single-agent/) | 7 (borough) / 261 (taxi zone) | 1 |
| Single-agent, zone-count sweep | [`single-agent-zone-sweep/`](single-agent-zone-sweep/) | 15 / 30 / 50 (KMeans) | 1 |
| Flat multi-agent (IPPO) | [`multi-agent/`](multi-agent/) | 7 (borough) | 7 |
| Flat multi-agent (IPPO) | [`multi-agent-10zone/`](multi-agent-10zone/) | 10 (KMeans) | 10 |
| Flat multi-agent (IPPO) | [`multi-agent-265zone/`](multi-agent-265zone/) | 261 (taxi zone) | 261 |
| Hierarchical, lead-first | [`hierarchical-lead-first/`](hierarchical-lead-first/) | 7 (borough) | 1 lead + 7 workers |
| Hierarchical, lead-first | [`hierarchical-lead-first-265zone/`](hierarchical-lead-first-265zone/) | 261 (taxi zone) | 1 lead + 260 workers |
| Hierarchical, workers-first | [`hierarchical-workers-first/`](hierarchical-workers-first/) | 7 (borough) | 1 lead + 7 workers |
| Hierarchical, workers-first | [`hierarchical-workers-first-265zone/`](hierarchical-workers-first-265zone/) | 261 (taxi zone) | 1 lead + 260 workers |

All twelve architecture/zone-count combinations are trained with Proximal
Policy Optimization (PPO, stable-baselines3 default MLP policy) and the
same equal-total-compute budget: 1,000,000 samples total, split across
however many learners that architecture uses. The two hierarchical
orderings differ only in which tier (lead or workers) trains first against
the other's frozen checkpoint each alternating round -- see each
`hierarchical-*` folder's README for the mechanics.

Each subfolder is self-contained: its own `README.md`, `requirements.txt`,
`config/config.yaml`, and `src/`. Raw trip data and trained model
checkpoints are not committed (`.gitignore`'d, regenerate via each
folder's data/training scripts) -- see **Data source** below.

## Quick start

### Single-agent (one PPO policy for all zones)
```bash
cd single-agent
pip install -r requirements.txt
python scripts/run_pipeline.py
```

> **NOTE:** RUN single agent first!! the other variants' data files are
> copied locally from it, not redownloaded :)

### Flat multi-agent (one agent per zone, shared policy, IPPO)
```bash
cd multi-agent
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.rl.train_ippo
python -m src.rl.evaluate_ippo
python -m src.rl.export_ippo_schedule
```

### Hierarchical (lead + workers, alternating best-response)
```bash
cd hierarchical-lead-first   # or hierarchical-workers-first
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.rl.train_hierarchical
python -m src.rl.evaluate_hierarchical
python -m src.rl.export_hierarchical_schedule
```

The `*-265zone` and `single-agent-zone-sweep` folders follow the same
commands, just with `config/config.yaml`'s `city.zone_level` /
`city.num_zones` set to the larger zone count.

## Data source

Raw data is NYC TLC yellow-taxi trip records and taxi zone lookups
(public), plus elder-care POIs from OpenStreetMap's Overpass API -- not
checked into this repo. Each variant's `src/data/download_*.py` scripts
fetch and cache it locally (`data/raw/`, `.gitignore`'d); `preprocess.py`
then builds the processed demand table each architecture trains against.
See [`single-agent/src/data/`](single-agent/src/data/) for the download
scripts (`download_taxi_zones.py`, `download_trip_data.py`,
`download_elder_care_pois.py`).
