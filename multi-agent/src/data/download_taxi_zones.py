"""Download the NYC TLC taxi zone lookup table and zone shapefile.

These give every zone's borough and a polygon geometry, which we use as the
spatial units for demand aggregation and fleet allocation.
"""

import zipfile
from pathlib import Path

import requests

from src.utils.config import load_config, resolve_path

ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_SHAPES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already present")
        return
    print(f"[download] {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def download_taxi_zones(cfg: dict) -> None:
    raw_dir = resolve_path(cfg, cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    lookup_path = raw_dir / "taxi_zone_lookup.csv"
    shapes_zip_path = raw_dir / "taxi_zones.zip"

    _download(ZONE_LOOKUP_URL, lookup_path)
    _download(ZONE_SHAPES_URL, shapes_zip_path)

    shapes_dir = raw_dir / "taxi_zones"
    if not shapes_dir.exists():
        with zipfile.ZipFile(shapes_zip_path) as zf:
            zf.extractall(shapes_dir)
        print(f"[extract] taxi_zones.zip -> {shapes_dir}")


if __name__ == "__main__":
    download_taxi_zones(load_config())
