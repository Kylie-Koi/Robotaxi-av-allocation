"""Download NYC TLC yellow taxi trip-record parquet files.

This is our public proxy for robotaxi demand: pickup/dropoff zone, timestamp,
and fare per trip. One file per year-month, e.g. "2024-01".
"""

from pathlib import Path

import requests

from src.utils.config import load_config, resolve_path

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download_month(year_month: str, raw_dir: Path) -> Path:
    fname = f"yellow_tripdata_{year_month}.parquet"
    dest = raw_dir / fname
    if dest.exists():
        print(f"[skip] {fname} already present")
        return dest

    url = f"{BASE_URL}/{fname}"
    print(f"[download] {url}")
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return dest


def download_all(cfg: dict) -> list[Path]:
    raw_dir = resolve_path(cfg, cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    return [download_month(ym, raw_dir) for ym in cfg["data"]["trip_months"]]


if __name__ == "__main__":
    download_all(load_config())
