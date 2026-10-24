"""Fetch elder-care facility locations (nursing homes, assisted living, etc.)
from OpenStreetMap via the public Overpass API, scoped to the target city.

Output: data/raw/elder_care_pois.csv with columns [name, lat, lon, tags].
"""

import csv
from pathlib import Path

import requests

from src.utils.config import load_config, resolve_path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Bounding box roughly covering NYC's five boroughs (south, west, north, east).
CITY_BBOX = {
    "nyc": (40.4957, -74.2557, 40.9176, -73.7002),
}


def _build_query(bbox: tuple[float, float, float, float], amenity_tags: list[str]) -> str:
    south, west, north, east = bbox
    clauses = []
    for tag in amenity_tags:
        key, value = tag.split("=", 1)
        clauses.append(f'node["{key}"="{value}"]({south},{west},{north},{east});')
        clauses.append(f'way["{key}"="{value}"]({south},{west},{north},{east});')
    body = "\n  ".join(clauses)
    return f"""
[out:json][timeout:120];
(
  {body}
);
out center;
""".strip()


def download_elder_care_pois(cfg: dict) -> Path:
    city = cfg["city"]["name"]
    bbox = CITY_BBOX.get(city)
    if bbox is None:
        raise ValueError(f"No bounding box configured for city '{city}'")

    query = _build_query(bbox, cfg["elder_care"]["amenity_tags"])
    print("[overpass] querying elder-care facilities...")
    resp = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers={"User-Agent": "robotaxi-av-allocation-research/0.1"},
        timeout=180,
    )
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    raw_dir = resolve_path(cfg, cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "elder_care_pois.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "lat", "lon", "tags"])
        for el in elements:
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat is None or lon is None:
                continue
            tags = el.get("tags", {})
            writer.writerow([tags.get("name", ""), lat, lon, tags])

    print(f"[saved] {len(elements)} elder-care facilities -> {out_path}")
    return out_path


if __name__ == "__main__":
    download_elder_care_pois(load_config())
