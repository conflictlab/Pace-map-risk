#!/usr/bin/env python3
"""
Download and prepare UCDP data for parallel forecast generation using public downloads
 - Base GED: https://ucdp.uu.se/downloads/ged/ged251-csv.zip
 - Candidate GED monthly CSVs: https://ucdp.uu.se/downloads/candidateged/GEDEvent_v{YY}_0_{i}.csv
"""
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import urllib.request
import tempfile

def http_download(url: str, suffix: str = '', timeout: int = 60, retries: int = 5, backoff_seconds: int = 5) -> str:
    """Download a URL to a temporary file with retries and return the local path."""
    last_err = None
    for attempt in range(retries):
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp_path, 'wb') as f:
                f.write(r.read())
            return tmp_path
        except Exception as e:
            last_err = e
            # Exponential-ish backoff
            time.sleep(backoff_seconds * (attempt + 1))
    raise last_err if last_err else RuntimeError(f"Failed to download: {url}")

print('Loading UCDP data from public downloads…')

# Determine ASOF month (YYYY-MM) if provided
ASOF = os.environ.get('ASOF')
try:
    NOW_DT = datetime.strptime(ASOF + '-01', '%Y-%m-%d') if ASOF else datetime.now()
except Exception:
    NOW_DT = datetime.now()

# 1) Base GED dataset (v25.1 ZIP) — fetch with retries to avoid transient timeouts
base_url = 'https://ucdp.uu.se/downloads/ged/ged251-csv.zip'
base_zip = http_download(base_url, suffix='.zip', timeout=90, retries=5, backoff_seconds=6)
df = pd.read_csv(base_zip, parse_dates=['date_start', 'date_end'], low_memory=False)

# 2) Candidate GED monthly CSVs for current GED stream (e.g., 2026 -> v26)
month = NOW_DT.strftime('%m')
if month == '01':
    month = '13'
major = int(NOW_DT.strftime('%y'))  # 2026 -> 26
print(f"Fetching Candidate GED monthly CSVs for v{major} (1..{int(month)-1})…")
def _read_candidate_csv(url: str) -> pd.DataFrame:
    """Robust CSV loader for UCDP candidate files with varying delimiters, with HTTP retries."""
    local = http_download(url, suffix='.csv', timeout=60, retries=5, backoff_seconds=5)
    for kwargs in (
        {},
        { 'sep': ';' },
        { 'engine': 'python' },
        { 'engine': 'python', 'sep': None },
    ):
        try:
            return pd.read_csv(local, **kwargs)
        except Exception:
            continue
    # If all variants fail, re-raise a clear error
    raise RuntimeError(f"Unable to parse candidate CSV: {url}")

for i in range(1, int(month)):
    url = f'https://ucdp.uu.se/downloads/candidateged/GEDEvent_v{major}_0_{i}.csv'
    try:
        df_can = _read_candidate_csv(url)
        # Align to base schema
        df_can.columns = df.columns
        df_can['date_start'] = pd.to_datetime(df_can['date_start'])
        df_can['date_end'] = pd.to_datetime(df_can['date_end'])
        df_can = df_can.drop_duplicates()
        df = pd.concat([df, df_can], axis=0)
        print(f"  ✓ Added candidate month {i} (v{major}_0_{i})")
    except Exception as e:
        print(f"  Note: Could not load {url}: {e}")

# 2b) Backfill previous year's candidates (full year)
prev_major = major - 1
print(f"Backfilling Candidate GED monthly CSVs for v{prev_major} (1..12)…")
for i in range(1, 13):
    url = f'https://ucdp.uu.se/downloads/candidateged/GEDEvent_v{prev_major}_0_{i}.csv'
    try:
        df_can = _read_candidate_csv(url)
        df_can.columns = df.columns
        df_can['date_start'] = pd.to_datetime(df_can['date_start'])
        df_can['date_end'] = pd.to_datetime(df_can['date_end'])
        df_can = df_can.drop_duplicates()
        df = pd.concat([df, df_can], axis=0)
        print(f"  ✓ Added backfill month {i} (v{prev_major}_0_{i})")
    except Exception as e:
        print(f"  Note: Could not load {url}: {e}")

# Normalize country names to standard set used downstream
country_name_mapping = {
    'DR Congo (Zaire)': 'Dem. Rep. Congo',
    'Myanmar (Burma)': 'Myanmar',
    'Russia (Soviet Union)': 'Russia',
    'South Sudan': 'S. Sudan',
    'Central African Republic': 'Central African Rep.',
    'Ivory Coast': 'Côte d\'Ivoire',
    'Kingdom of eSwatini (Swaziland)': 'eSwatini',
    'Dominican Republic': 'Dominican Rep.',
    'Macedonia, FYR': 'Macedonia',
    'Madagascar (Malagasy)': 'Madagascar',
    'North Macedonia': 'Macedonia',
    'Serbia (Yugoslavia)': 'Serbia',
    'Yemen (North Yemen)': 'Yemen',
    'Zimbabwe (Rhodesia)': 'Zimbabwe',
    'Vietnam (North Vietnam)': 'Vietnam'
}
if 'country' in df.columns:
    df['country'] = df['country'].replace(country_name_mapping)

print('\nProcessing data…')
df_tot = pd.DataFrame(columns=df.country.unique(),
                     index=pd.date_range(df.date_start.min(), df.date_end.max()))
df_tot = df_tot.fillna(0)

for i in df.country.unique():
    df_sub = df[df.country == i]
    for j in range(len(df_sub)):
        if df_sub.date_start.iloc[j].month == df_sub.date_end.iloc[j].month:
            df_tot.loc[df_sub.date_start.iloc[j], i] = \
                df_tot.loc[df_sub.date_start.iloc[j], i] + df_sub.best.iloc[j]

df_tot_m = df_tot.resample('M').sum()

# Cut off at last complete month relative to ASOF or now
last_month = NOW_DT.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
df_tot_m = df_tot_m.loc[:last_month, :]

# Drop trailing months with all zeros (UCDP data not yet released)
while len(df_tot_m) > 0 and df_tot_m.iloc[-1].sum() == 0:
    dropped_month = df_tot_m.index[-1]
    print(f'⚠️  Dropping {dropped_month.strftime("%Y-%m")} (all zeros - UCDP data not released)')
    df_tot_m = df_tot_m.iloc[:-1]

# Save to CSV
df_tot_m.to_csv('Conf.csv')

print(f'\nProcessed {len(df_tot_m.columns)} countries')
print(f'Data span: {df_tot_m.index[0].strftime("%Y-%m")} to {df_tot_m.index[-1].strftime("%Y-%m")}')
print('Data preparation complete!')
