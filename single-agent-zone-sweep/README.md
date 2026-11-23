# Robotaxi / AV Fleet Allocation -- Single-Agent Zone-Count Sweep (K=15/30/50)

Same single-agent architecture and objective as [single-agent](../single-agent):
forecasts robotaxi demand in a city and optimizes fleet allocation with
reinforcement learning, maximizing profit subject to a service-level
constraint that prioritizes elder-care locations (nursing homes, assisted
living facilities). This variant isolates the effect of **zone count**
(decomposition granularity) independent of architecture, by running the
single-agent policy at `config.yaml`'s `city.zone_level: kmeans` with
`city.num_zones` set to 15, 30, and 50 geographic KMeans clusters of the
261 active taxi zones -- in between the 7-borough and full 261-taxi-zone
single-agent runs. Each zone count was trained as a separate run (same
`config.yaml`, `num_zones` changed between runs).

## Pipeline

1. **Data** (`src/data/`)
   - `download_taxi_zones.py` — NYC TLC taxi zone lookup + shapefile (spatial units)
   - `download_trip_data.py` — NYC TLC yellow taxi trip records (public demand proxy)
   - `download_elder_care_pois.py` — nursing homes / assisted living, via OpenStreetMap Overpass API
   - `preprocess.py` — aggregates trips into a (zone/borough, time bin) demand table and flags elder-care units
2. **Forecasting** (`src/forecasting/`)
   - `models.py` — `SeasonalNaiveForecaster` (baseline) and `LSTMForecaster` (learned), common interface
   - `train.py` / `evaluate.py` — fit and score on a held-out time split
3. **Simulation + RL** (`src/simulation/`, `src/rl/`)
   - `fleet_env.py` — Gymnasium env: agent reallocates the fleet across zones each time step; reward = fare revenue − repositioning cost − elder-care shortfall penalty
   - `train_agent.py` — trains PPO (stable-baselines3)
   - `evaluate_agent.py` — compares the trained policy against uniform and demand-proportional baselines

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_pipeline.py
```

Or step by step:

```bash
python -m src.data.download_taxi_zones
python -m src.data.download_trip_data
python -m src.data.download_elder_care_pois
python -m src.data.preprocess
python -m src.forecasting.evaluate
python -m src.rl.train_agent
python -m src.rl.evaluate_agent
```

## Configuration

All knobs (city, zone granularity, months of data, fleet size, elder-care
service level, RL hyperparameters) live in `config/config.yaml`.

Key choices to revisit as the project matures:

- `city.zone_level`: `borough` (default, ~6 units, fast to prototype) vs
  `taxi_zone` (265 units, finer-grained but a much larger action space)
- `elder_care.min_service_level` / `min_vehicles`: how strictly elder-care
  zones are protected — currently enforced as a reward penalty rather than a
  hard constraint, so the agent can still technically violate it under
  extreme demand; consider action-masking or a Lagrangian/CMDP formulation
  if strict guarantees are required
- `forecasting.model`: swap `seasonal_naive` → `lstm` once there's enough
  historical data per zone to train it well

## Notes on the current simplifications

- Vehicles are treated as a continuous quantity per zone, not discrete units
  with individual positions/routes — fine for an allocation-level policy,
  but won't capture per-vehicle dispatch/routing.
- The RL env uses ground-truth historical demand as the "actual" outcome for
  reward calculation, and the forecaster's prediction only as the agent's
  *observation* — this mirrors a realistic setting where allocation decisions
  are made on a forecast but scored against what actually happens.
- NYC yellow-taxi trips are a proxy for robotaxi demand; fare structure and
  trip patterns differ from an actual AV service, so profit numbers are
  illustrative, not predictive.
