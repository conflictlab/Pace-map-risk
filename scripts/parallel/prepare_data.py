#!/usr/bin/env python3
"""
Download and prepare UCDP data for parallel forecast generation using UCDP API
"""
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
import sys

# Get API token from environment variable
API_TOKEN = os.environ.get('UCDP_API_TOKEN')
if not API_TOKEN:
    print('ERROR: UCDP_API_TOKEN environment variable not set')
    print('Please set your UCDP API token as an environment variable')
    sys.exit(1)

def fetch_ucdp_api(version, pagesize=1000):
    """Fetch UCDP GED data from API with pagination"""
    base_url = f'https://ucdpapi.pcr.uu.se/api/gedevents/{version}'
    headers = {'x-ucdp-access-token': API_TOKEN}

    all_events = []
    page = 0

    print(f'Fetching UCDP GED version {version}...')

    while True:
        params = {'pagesize': pagesize, 'page': page}
        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code != 200:
            print(f'Error fetching page {page}: HTTP {response.status_code}')
            break

        data = response.json()
        events = data.get('Result', [])

        if not events:
            break

        all_events.extend(events)

        total_pages = data.get('TotalPages', 0)
        print(f'  Downloaded page {page + 1}/{total_pages} ({len(all_events)} events so far)', end='\r')

        page += 1

        # Stop if we've reached the last page
        if page >= total_pages:
            break

    print(f'\n  Total events downloaded: {len(all_events)}')
    return pd.DataFrame(all_events)

print('Loading UCDP data via API...')

# Fetch main GED dataset (version 25.1 - covers 1989-2024)
df_main = fetch_ucdp_api('25.1')

# Fetch latest candidate data (version 26.0.1 - monthly release with most recent data)
print('\nFetching latest candidate data (version 26.0.1)...')
try:
    df_candidate = fetch_ucdp_api('26.0.1')
    print(f'  Candidate data: {len(df_candidate)} events')

    # Combine datasets
    df = pd.concat([df_main, df_candidate], axis=0)
    df = df.drop_duplicates(subset=['id'], keep='last')
    print(f'\nCombined dataset: {len(df)} unique events')
except Exception as e:
    print(f'Note: Could not load candidate data v26.0.1: {e}')
    print('Using main dataset only (v25.1)')
    df = df_main

# Convert date columns to datetime
df['date_start'] = pd.to_datetime(df['date_start'])
df['date_end'] = pd.to_datetime(df['date_end'])

print('\nProcessing data...')
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

# Cut off at last complete month
last_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
df_tot_m = df_tot_m.loc[:last_month, :]

# Save to CSV
df_tot_m.to_csv('Conf.csv')

print(f'\nProcessed {len(df_tot_m.columns)} countries')
print(f'Data span: {df_tot_m.index[0].strftime("%Y-%m")} to {df_tot_m.index[-1].strftime("%Y-%m")}')
print('Data preparation complete!')
