#!/usr/bin/env python3
"""
Create Hist.csv and metadata files
"""
import pandas as pd
import json
from datetime import datetime, timedelta
import os

def rename_countries(df):
    return df.rename(columns={
        'Bosnia-Herzegovina': 'Bosnia and Herz.',
        'Cambodia (Kampuchea)': 'Cambodia',
        'Central African Republic': 'Central African Rep.',
        'DR Congo (Zaire)': 'Dem. Rep. Congo',
        'Ivory Coast': "Côte d'Ivoire",
        'Kingdom of eSwatini (Swaziland)': 'eSwatini',
        'Dominican Republic': 'Dominican Rep.',
        'Macedonia, FYR': 'Macedonia',
        'Madagascar (Malagasy)': 'Madagascar',
        'Myanmar (Burma)': 'Myanmar',
        'North Macedonia': 'Macedonia',
        'Russia (Soviet Union)': 'Russia',
        'Serbia (Yugoslavia)': 'Serbia',
        'South Sudan': 'S. Sudan',
        'Yemen (North Yemen)': 'Yemen',
        'Zimbabwe (Rhodesia)': 'Zimbabwe',
        'Vietnam (North Vietnam)': 'Vietnam'
    })

# Create Hist.csv
print('Creating Hist.csv...')
df_tot_m = pd.read_csv('data/Conf.csv', index_col=0, parse_dates=True)

# Drop trailing months with all zeros (UCDP data not yet released)
while len(df_tot_m) > 0 and df_tot_m.iloc[-1].sum() == 0:
    dropped_month = df_tot_m.index[-1]
    print(f'⚠️  Dropping {pd.Timestamp(dropped_month).strftime("%Y-%m")} (all zeros - UCDP data not released)')
    df_tot_m = df_tot_m.iloc[:-1]

hist_full = rename_countries(df_tot_m)
hist_full.to_csv('Hist.csv')
print(f'Saved Hist.csv with {len(hist_full)} months of data')

# Create metadata (use actual data end from cleaned df_tot_m)
data_end = pd.Timestamp(df_tot_m.index[-1])
metadata = {
    'run_date': datetime.now().isoformat(),
    'data_end_date': data_end.strftime('%Y-%m'),
    'forecast_start_date': (data_end + pd.DateOffset(months=1)).strftime('%Y-%m'),
    'h6_end_date': (data_end + pd.DateOffset(months=6)).strftime('%Y-%m'),
    'h12_end_date': (data_end + pd.DateOffset(months=12)).strftime('%Y-%m'),
    'training_window_months': 10,
    'historical_start_date': df_tot_m.index[0].strftime('%Y-%m'),
    'historical_end_date': data_end.strftime('%Y-%m'),
    'total_historical_months': len(df_tot_m),
    'parallelization': 'Matrix strategy with 4 chunks per horizon'
}

with open('forecast_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('Created forecast_metadata.json')
