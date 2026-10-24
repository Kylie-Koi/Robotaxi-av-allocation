"""Run the full pipeline end-to-end: download -> preprocess -> forecast -> train RL -> evaluate.

Usage:
    python scripts/run_pipeline.py [--skip-download]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download_taxi_zones import download_taxi_zones
from src.data.download_trip_data import download_all
from src.data.download_elder_care_pois import download_elder_care_pois
from src.data.preprocess import build_demand_table
from src.forecasting import evaluate as forecast_evaluate
from src.rl.train_agent import train_agent
from src.rl.evaluate_agent import main as evaluate_agent_main
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    if not args.skip_download:
        download_taxi_zones(cfg)
        download_all(cfg)
        download_elder_care_pois(cfg)

    build_demand_table(cfg)
    forecast_evaluate.main()
    train_agent(cfg)
    evaluate_agent_main()


if __name__ == "__main__":
    main()
