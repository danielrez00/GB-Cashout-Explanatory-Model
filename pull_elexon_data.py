"""
Pull GB electricity market data from the Elexon BMRS API and cache it as parquet.

Sources (all half-hourly settlement-period resolution):
    system prices    /balancing/settlement/system-prices    cash-out price, NIV
    demand           /demand/outturn                         initial transmission demand outturn
    wind generation  /generation/outturn/summary             actual wind by fuel type
    wind forecast    /forecast/generation/wind/earliest      earliest published WINDFOR

Outputs, each indexed by UTC timestamp:
    system_prices_2024.parquet
    demand_2024.parquet
    wind_2024.parquet
    forecast_2024.parquet

Vintage note: price and NIV are near-settled values, not point-in-time, because
BMRS revises settlement data through reconciliation runs. The wind forecast uses
the 'earliest' endpoint, which returns the first published forecast for each
period and is therefore ex-ante (no look-ahead).

Usage:
    python pull_elexon_data.py
"""

import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm

BASE = "https://data.elexon.co.uk/bmrs/api/v1"
TIMEOUT = 30


# ---------------------------------------------------------------------------
# Endpoint fetchers
# ---------------------------------------------------------------------------
def _iso(d):
    """Accept a date/datetime or a string and return an ISO string."""
    return d.isoformat() if hasattr(d, "isoformat") else d


def get_system_prices_one_date(settlement_date):
    """Pull all settlement periods (46/48/50) for a single date."""
    url = f"{BASE}/balancing/settlement/system-prices/{_iso(settlement_date)}"
    r = requests.get(url, params={"format": "json"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["data"]


def get_wind_forecast(start_date, end_date):
    """Earliest published wind forecast (WINDFOR) over an inclusive date range."""
    url = f"{BASE}/forecast/generation/wind/earliest"
    params = {"from": _iso(start_date), "to": _iso(end_date), "format": "json"}
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["data"]


def get_demand_range(start_date, end_date):
    """Initial transmission system demand outturn over an inclusive date range."""
    url = f"{BASE}/demand/outturn"
    params = {
        "settlementDateFrom": _iso(start_date),
        "settlementDateTo": _iso(end_date),
        "format": "json",
    }
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["data"]


def get_generation_outturn(start_time, end_time, include_negative=False):
    """
    Generation outturn summary (by fuel type) over a continuous datetime range.
    This endpoint returns a bare list of period-records, not a {'data': ...} wrapper.
    """
    url = f"{BASE}/generation/outturn/summary"
    if isinstance(start_time, datetime):
        start_time = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(end_time, datetime):
        end_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "startTime": start_time,
        "endTime": end_time,
        "includeNegativeGeneration": str(include_negative).lower(),
        "format": "json",
    }
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Range drivers
#
# Three endpoints, three range conventions:
#   - system prices take a single date in the path, so we loop day by day.
#   - demand and wind forecast take an inclusive settlementDate range, tiled
#     with no overlap by pull_date_chunks.
#   - generation outturn takes a continuous datetime range, tiled with a shared
#     boundary by pull_time_chunks.
# ---------------------------------------------------------------------------
def pull_system_prices(start_date, end_date, sleep=0.1):
    """Loop the per-date system-prices endpoint over an inclusive date range."""
    all_rows = []
    n_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    for current in tqdm(dates, desc="system prices"):
        try:
            all_rows.extend(get_system_prices_one_date(current))
        except requests.HTTPError as e:
            tqdm.write(f"  {current} FAILED: {e}")
        time.sleep(sleep)
    return all_rows


def pull_date_chunks(start_date, end_date, fetch_fn, chunk_days=28, sleep=0.2):
    """Tile an inclusive settlementDate range into non-overlapping chunks."""
    all_rows = []
    current = start_date
    with tqdm(desc=fetch_fn.__name__) as pbar:
        while current <= end_date:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
            try:
                rows = fetch_fn(current, chunk_end)
                all_rows.extend(rows)
                tqdm.write(f"  [{current} -> {chunk_end}] {len(rows)} rows")
            except requests.HTTPError as e:
                tqdm.write(f"  [{current} -> {chunk_end}] FAILED: {e}")
            current = chunk_end + timedelta(days=1)
            pbar.update(1)
            time.sleep(sleep)
    return all_rows


def pull_time_chunks(start_date, end_date, fetch_fn, chunk_days=7, sleep=0.2):
    """Tile a continuous datetime range into chunks sharing their boundaries."""
    all_rows = []
    current = start_date
    with tqdm(desc=fetch_fn.__name__) as pbar:
        while current < end_date:
            chunk_end = min(current + timedelta(days=chunk_days), end_date)
            try:
                rows = fetch_fn(current, chunk_end)
                all_rows.extend(rows)
                tqdm.write(f"  [{current} -> {chunk_end}] {len(rows)} periods")
            except requests.HTTPError as e:
                tqdm.write(f"  [{current} -> {chunk_end}] FAILED: {e}")
            current = chunk_end
            pbar.update(1)
            time.sleep(sleep)
    return all_rows


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------
def sp_to_datetime(settlement_date, settlement_period):
    """
    Convert (date, settlement period 1-50) to a tz-aware UTC timestamp.
    Settlement periods are defined in local time, so we build the timestamp in
    Europe/London and convert. Clock-change days are handled by the localize
    flags: the redundant autumn period maps to NaT and is dropped downstream.
    """
    local_midnight = pd.Timestamp(settlement_date).tz_localize(
        "Europe/London", nonexistent="shift_forward", ambiguous="NaT"
    )
    return (
        local_midnight + pd.Timedelta(minutes=30 * (settlement_period - 1))
    ).tz_convert("UTC")


def build_sp_dataframe(rows):
    """
    Build a UTC-indexed frame from settlement-period records. Used for system
    prices, demand, and the wind forecast, which all carry settlementDate and
    settlementPeriod fields.
    """
    df = pd.DataFrame(rows)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"]).dt.date
    df["timestamp"] = [
        sp_to_datetime(d, p)
        for d, p in zip(df["settlementDate"], df["settlementPeriod"])
    ]
    df = df.dropna(subset=["timestamp"])  # drops the ambiguous autumn DST period
    return df.set_index("timestamp").sort_index()


def build_generation_dataframe(rows, fuels=("WIND",)):
    """
    Flatten nested generation outturn into one column per requested fuel type,
    indexed by timestamp. Generation records are keyed by startTime, not by
    settlement period, so they use their own builder.
    """
    records = []
    for period in rows:
        fuel_map = {f["fuelType"]: f["generation"] for f in period["data"]}
        record = {"timestamp": period["startTime"]}
        for fuel in fuels:
            record[fuel.lower()] = fuel_map.get(fuel)  # None if absent that period
        records.append(record)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    START = date(2024, 1, 1)
    END = date(2024, 12, 31)

    print("Pulling wind forecast...")
    forecast_rows = pull_date_chunks(START, END, get_wind_forecast)

    print("Pulling system prices...")
    price_rows = pull_system_prices(START, END)

    print("Pulling wind generation...")
    wind_rows = pull_time_chunks(START, END, get_generation_outturn)

    print("Pulling system demand...")
    demand_rows = pull_date_chunks(START, END, get_demand_range)

    frames = {
        "forecast_2024": build_sp_dataframe(forecast_rows),
        "system_prices_2024": build_sp_dataframe(price_rows),
        "demand_2024": build_sp_dataframe(demand_rows),
        "wind_2024": build_generation_dataframe(wind_rows),
    }

    for name, frame in frames.items():
        print(f"  {name}: {frame.shape}, columns {frame.columns.tolist()}")
        frame.to_parquet(f"{name}.parquet")

    print("Saved all parquet files.")
