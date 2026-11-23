"""Build the per-zone, per-time-bin demand table used by forecasting and the RL env.

Steps:
  1. Load taxi zone lookup (+ shapes) and pick the spatial unit (zone or borough).
  2. Load trip parquet files, aggregate pickups into (unit, time_bin) counts,
     avg fare, and avg trip duration.
  3. Spatially join elder-care POIs onto units and flag which units contain one.

Output: data/processed/demand.parquet with columns
  [unit_id, unit_name, timestamp, trip_count, avg_fare, avg_trip_duration_minutes, is_elder_care]

avg_trip_duration_minutes matters beyond the fare: it sets how many trips one
vehicle can actually complete per hour (a vehicle tied up on a 45-minute trip
can't also serve a second rider that hour), which the simulation environments
use to convert a raw vehicle count into effective hourly capacity.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.utils.config import load_config, resolve_path


def _load_zone_lookup(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "taxi_zone_lookup.csv")


def _load_zone_shapes(raw_dir: Path) -> gpd.GeoDataFrame:
    shp_dir = raw_dir / "taxi_zones"
    shp_files = list(shp_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found under {shp_dir}")
    gdf = gpd.read_file(shp_files[0])
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def _build_unit_table(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """Return a mapping LocationID -> unit_id, unit_name at the configured zone_level."""
    lookup = _load_zone_lookup(raw_dir)
    lookup["Borough"] = lookup["Borough"].fillna("Unknown")
    zone_level = cfg["city"]["zone_level"]

    if zone_level == "taxi_zone":
        lookup["unit_id"] = lookup["LocationID"]
        lookup["unit_name"] = lookup["Zone"]
    elif zone_level == "borough":
        boroughs = sorted(lookup["Borough"].unique())
        borough_to_id = {b: i for i, b in enumerate(boroughs)}
        lookup["unit_id"] = lookup["Borough"].map(borough_to_id)
        lookup["unit_name"] = lookup["Borough"]
    elif zone_level == "kmeans":
        # Groups the 265 taxi zones into cfg["city"]["num_zones"] geographic
        # clusters (by zone centroid), for a granularity between "borough"
        # (7, coarse) and "taxi_zone" (261, fine) -- ported from the
        # multiagent_10zones sibling project's preprocess.py.
        num_zones = cfg["city"]["num_zones"]
        shapes = _load_zone_shapes(raw_dir)
        # NY State Plane (feet) -- a projected CRS so centroid/distance are
        # accurate, unlike computing centroids directly in lat/lon degrees.
        centroids = shapes.geometry.to_crs(epsg=2263).centroid
        coords = np.column_stack([centroids.x, centroids.y])
        labels = KMeans(n_clusters=num_zones, random_state=0, n_init=10).fit_predict(coords)
        cluster_by_location = dict(zip(shapes["LocationID"], labels))
        lookup["unit_id"] = lookup["LocationID"].map(cluster_by_location)
        lookup["unit_name"] = lookup["unit_id"].apply(lambda i: f"Cluster {int(i)}" if pd.notna(i) else None)
    else:
        raise ValueError(f"Unknown zone_level: {zone_level}")

    return lookup[["LocationID", "unit_id", "unit_name"]]


def _aggregate_trips(cfg: dict, raw_dir: Path, unit_table: pd.DataFrame) -> pd.DataFrame:
    bin_minutes = cfg["data"]["time_bin_minutes"]
    parquet_files = sorted(raw_dir.glob("yellow_tripdata_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No trip parquet files found in {raw_dir}")

    frames = []
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "fare_amount"]
    for pf in parquet_files:
        print(f"[load] {pf.name}")
        df = pd.read_parquet(pf, columns=cols)

        # TLC trip files contain a handful of corrupted pickup timestamps
        # (sometimes decades off); drop rows outside the file's own month so
        # they don't blow up the later date_range reindex.
        year_month = pf.stem.rsplit("_", 1)[-1]  # e.g. "2024-01"
        month_start = pd.Timestamp(year_month)
        month_end = month_start + pd.DateOffset(months=1)
        df = df[(df["tpep_pickup_datetime"] >= month_start) & (df["tpep_pickup_datetime"] < month_end)]

        df["trip_duration_minutes"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60.0
        # Drop implausibly short or long durations (data errors, not real
        # trips -- a real taxi ride takes at least a minute) before they
        # skew the per-zone average, especially in sparse low-demand zones
        # where a single artifact can dominate the mean.
        df = df[(df["trip_duration_minutes"] >= 1) & (df["trip_duration_minutes"] <= 180)]

        df = df.merge(unit_table, left_on="PULocationID", right_on="LocationID", how="inner")
        df["timestamp"] = df["tpep_pickup_datetime"].dt.floor(f"{bin_minutes}min")
        frames.append(df[["unit_id", "unit_name", "timestamp", "fare_amount", "trip_duration_minutes"]])

    all_trips = pd.concat(frames, ignore_index=True)
    all_trips = all_trips[all_trips["fare_amount"] > 0]

    grouped = (
        all_trips.groupby(["unit_id", "unit_name", "timestamp"])
        .agg(
            trip_count=("fare_amount", "count"),
            avg_fare=("fare_amount", "mean"),
            avg_trip_duration_minutes=("trip_duration_minutes", "mean"),
        )
        .reset_index()
    )
    return grouped


def _flag_elder_care_units(cfg: dict, raw_dir: Path, unit_table: pd.DataFrame) -> set[int]:
    poi_path = raw_dir / "elder_care_pois.csv"
    if not poi_path.exists():
        print("[warn] elder_care_pois.csv not found; no units will be flagged")
        return set()

    pois = pd.read_csv(poi_path)
    pois = pois.dropna(subset=["lat", "lon"])
    poi_gdf = gpd.GeoDataFrame(
        pois, geometry=gpd.points_from_xy(pois["lon"], pois["lat"]), crs="EPSG:4326"
    )

    zone_shapes = _load_zone_shapes(raw_dir)
    zone_shapes = zone_shapes.merge(
        unit_table, left_on="LocationID", right_on="LocationID", how="left"
    )

    joined = gpd.sjoin(poi_gdf, zone_shapes[["unit_id", "geometry"]], how="inner", predicate="within")
    return set(joined["unit_id"].dropna().unique())


def build_demand_table(cfg: dict) -> pd.DataFrame:
    raw_dir = resolve_path(cfg, cfg["data"]["raw_dir"])
    processed_dir = resolve_path(cfg, cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    unit_table = _build_unit_table(cfg, raw_dir)
    demand = _aggregate_trips(cfg, raw_dir, unit_table)

    elder_care_units = _flag_elder_care_units(cfg, raw_dir, unit_table)
    demand["is_elder_care"] = demand["unit_id"].isin(elder_care_units)

    # Fill missing (unit, timestamp) combinations with zero demand so every
    # unit has a complete, regularly-spaced time series.
    full_index = pd.MultiIndex.from_product(
        [demand["unit_id"].unique(), pd.date_range(demand["timestamp"].min(), demand["timestamp"].max(),
                                                     freq=f"{cfg['data']['time_bin_minutes']}min")],
        names=["unit_id", "timestamp"],
    )
    demand = (
        demand.set_index(["unit_id", "timestamp"])
        .reindex(full_index)
        .reset_index()
    )
    demand["trip_count"] = demand["trip_count"].fillna(0)
    demand["avg_fare"] = demand.groupby("unit_id")["avg_fare"].transform(lambda s: s.fillna(s.mean()))
    city_avg_duration = demand["avg_trip_duration_minutes"].mean()
    demand["avg_trip_duration_minutes"] = (
        demand.groupby("unit_id")["avg_trip_duration_minutes"].transform(lambda s: s.fillna(s.mean()))
    )
    # A handful of near-zero-demand units (e.g. EWR, Staten Island) may have
    # no valid trips at all in some periods; fall back to the city-wide average
    # rather than leaving capacity undefined for those hours.
    demand["avg_trip_duration_minutes"] = demand["avg_trip_duration_minutes"].fillna(city_avg_duration)
    name_map = demand.dropna(subset=["unit_name"]).groupby("unit_id")["unit_name"].first()
    elder_map = demand.dropna(subset=["is_elder_care"]).groupby("unit_id")["is_elder_care"].first()
    demand["unit_name"] = demand["unit_id"].map(name_map)
    demand["is_elder_care"] = demand["unit_id"].map(elder_map).fillna(False)

    out_path = processed_dir / "demand.parquet"
    demand.to_parquet(out_path, index=False)
    print(f"[saved] demand table -> {out_path} ({len(demand)} rows, {demand['unit_id'].nunique()} units)")
    return demand


if __name__ == "__main__":
    build_demand_table(load_config())
