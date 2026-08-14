#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Downloads the Cour de cassation ruling corpus from the Judilibre API (PISTE).

Each decision is saved as its own file, then consolidated into two tables:
the full corpus, and a cleaned copy keeping only rulings with a real appeal
number (pourvoi). An existing Judilibre dump can be pointed to directly
instead, skipping this script. Requires a PISTE API key in an environment
variable. The download pages through each time window in full and skips
decisions already on disk, so an interrupted run resumes where it left off.
"""
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import load_config, resolve  # noqa: E402

_cfg = load_config()
_collect_cfg = _cfg["collect"]

API_KEY_ENV = _collect_cfg["api_key_env"]
API_KEY = os.environ.get(API_KEY_ENV)

RAW_DECISIONS_DIR = resolve(_cfg["data"]["raw_decisions_dir"])
RAW_PICKLE = resolve(_cfg["data"]["raw_pickle"])
CLEAN_PICKLE = resolve(_cfg["data"]["clean_pickle"])

API_BASE_URL = _collect_cfg["api_base_url"]
JURISDICTION = _collect_cfg["jurisdiction"]
WINDOW_DAYS = _collect_cfg["window_days"]
BATCH_SIZE = _collect_cfg["batch_size"]
DATE_MIN = date.fromisoformat(_collect_cfg["date_min"])
MAX_CONSECUTIVE_EMPTY = _collect_cfg["max_consecutive_empty_windows"]


def fetch_batch(date_start_iso, date_end_iso, batch):
    """One API call, with retries."""
    url = (
        f"{API_BASE_URL}"
        f"?jurisdiction={JURISDICTION}&date_start={date_start_iso}&date_end={date_end_iso}"
        f"&batch_size={BATCH_SIZE}&batch={batch}"
    )
    headers = {"KeyId": API_KEY}
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, timeout=120)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = 10 * (attempt + 1)
                print(f"  error {r.status_code}, waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  exception (attempt {attempt+1}): {e}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed on {date_start_iso} -> {date_end_iso} batch {batch}")


def collect_decisions():
    """Main loop: walk backwards in time by WINDOW_DAYS-day windows, one JSON
    file per decision written to RAW_DECISIONS_DIR."""
    if not API_KEY:
        raise SystemExit(
            f"Missing PISTE API key: set the {API_KEY_ENV} environment variable "
            "(see README.md for how to request a key)."
        )
    RAW_DECISIONS_DIR.mkdir(parents=True, exist_ok=True)

    delta = timedelta(days=WINDOW_DAYS)
    date_end = date.today()
    date_end_iso = date_end.isoformat()
    date_start = date_end - delta
    date_start_iso = date_start.isoformat()

    consecutive_empty = 0

    while date_end > DATE_MIN and consecutive_empty < MAX_CONSECUTIVE_EMPTY:
        batch = 0
        n_in_window = 0
        while True:
            res = fetch_batch(date_start_iso, date_end_iso, batch)
            results = res.get("results", [])
            if not results:
                break
            n_in_window += len(results)
            # No filter on type here: from 2016 (the "J21" reform, Justice du
            # 21e siècle), the social
            # chamber mostly renders under type='other', not 'arret'. A naive
            # type='arret' filter would drop ~95% of post-2016 soc decisions.
            # 02_zone_and_pair.py filters on chamber/year/solution instead.
            for item in results:
                identifiant = item["id"]
                path = RAW_DECISIONS_DIR / f"{identifiant}.json"
                if path.exists():
                    continue
                path.write_text(json.dumps(item, ensure_ascii=False), encoding="utf8")
            if len(results) < BATCH_SIZE:
                break
            batch += 1
            time.sleep(0.2)

        print(f"{date_start_iso} -> {date_end_iso}: {n_in_window} decisions ({batch+1} batch{'es' if batch else ''})")

        consecutive_empty = 0 if n_in_window else consecutive_empty + 1

        date_end = date_start
        date_end_iso = date_end.isoformat()
        date_start = date_end - delta
        date_start_iso = date_start.isoformat()

    print("Done collecting.")


def consolidate_pickles():
    """Load every downloaded JSON into one DataFrame, and write the raw and
    clean pickles. Nested fields (themes, visa, publication, and so on) are
    kept as-is. A gzip-compressed pickle handles them natively."""
    data = []
    files_list = [f for f in os.listdir(RAW_DECISIONS_DIR) if f.endswith(".json")]

    for fname in tqdm(files_list, desc="Loading JSONs"):
        with open(RAW_DECISIONS_DIR / fname, encoding="utf8") as f:
            j = json.load(f)
        if not j.get("text"):
            continue
        row = dict(j)
        row["has_summary"] = "summary" in j and bool(j.get("summary"))
        data.append(row)

    df = pd.DataFrame(data)
    print(f"{len(df):,} decisions loaded, {len(df.columns)} columns")

    RAW_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(RAW_PICKLE, compression="gzip")
    print(f"\n{RAW_PICKLE.name}: {len(df):,} decisions, {RAW_PICKLE.stat().st_size / 1e9:.2f} GB")

    # keep only decisions with a real pourvoi number
    df = df[df["number"].notna() & (df["number"] != "-.")].reset_index(drop=True)
    CLEAN_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CLEAN_PICKLE, compression="gzip")
    print(f"{CLEAN_PICKLE.name}: {len(df):,} decisions, {CLEAN_PICKLE.stat().st_size / 1e9:.2f} GB")


def main():
    argv = set(sys.argv[1:])
    if "--skip-download" not in argv:
        collect_decisions()
    if "--skip-consolidate" not in argv:
        consolidate_pickles()


if __name__ == "__main__":
    main()
