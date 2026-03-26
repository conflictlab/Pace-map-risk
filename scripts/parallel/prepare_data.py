#!/usr/bin/env python3
"""
Download and prepare UCDP data for parallel forecast generation using public downloads
 - Base GED: https://ucdp.uu.se/downloads/ged/ged251-csv.zip
 - Candidate GED monthly CSVs: https://ucdp.uu.se/downloads/candidateged/GEDEvent_v{YY}_0_{i}.csv
"""
import pandas as pd
from datetime import datetime, timedelta
import os

print('Loading UCDP data from public downloads…')

# Determine ASOF month (YYYY-MM) if provided
ASOF = os.environ.get('ASOF')
try:
    NOW_DT = datetime.strptime(ASOF + '-01', '%Y-%m-%d') if ASOF else datetime.now()
except Exception:
    NOW_DT = datetime.now()

# 1) Base GED dataset (v25.1 ZIP)
df = pd.read_csv(
    'https://ucdp.uu.se/downloads/ged/ged251-csv.zip',
    parse_dates=['date_start', 'date_end'],
    low_memory=False
)

# 2) Candidate GED monthly CSVs for current GED stream (e.g., 2026 -> v26)
month = NOW_DT.strftime('%m')
if month == '01':
    month = '13'
major = int(NOW_DT.strftime('%y'))  # 2026 -> 26
print(f"Fetching Candidate GED monthly CSVs for v{major} (1..{int(month)-1})…")
def _read_candidate_csv(url: str) -> pd.DataFrame:
    """Robust CSV loader for UCDP candidate files with varying delimiters."""
    for kwargs in (
        {},
        { 'sep': ';' },
        { 'engine': 'python' },
        { 'engine': 'python', 'sep': None },
    ):
        try:
            return pd.read_csv(url, **kwargs)
        except Exception:
            continue
    raise

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

# Save to CSV
df_tot_m.to_csv('Conf.csv')

print(f'\nProcessed {len(df_tot_m.columns)} countries')
print(f'Data span: {df_tot_m.index[0].strftime("%Y-%m")} to {df_tot_m.index[-1].strftime("%Y-%m")}')
print('Data preparation complete!')
