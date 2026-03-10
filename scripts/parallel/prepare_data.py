#!/usr/bin/env python3
"""
Download and prepare UCDP data for parallel forecast generation
"""
import pandas as pd
from datetime import datetime, timedelta
import sys

print('Loading UCDP data...')
df = pd.read_csv('https://ucdp.uu.se/downloads/ged/ged251-csv.zip',
                 parse_dates=['date_start', 'date_end'], low_memory=False)
month = datetime.now().strftime('%m')

if month == '01':
    month = '13'
for i in range(1, int(month)):
    try:
        df_can = pd.read_csv(f'https://ucdp.uu.se/downloads/candidateged/GEDEvent_v25_0_{i}.csv')
        df_can.columns = df.columns
        df_can['date_start'] = pd.to_datetime(df_can['date_start'])
        df_can['date_end'] = pd.to_datetime(df_can['date_end'])
        df_can = df_can.drop_duplicates()
        df = pd.concat([df, df_can], axis=0)
    except Exception as e:
        print(f'Note: Could not load candidate data for month {i}: {e}')

print('Processing data...')
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
last_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
df_tot_m = df_tot_m.loc[:last_month, :]
df_tot_m.to_csv('Conf.csv')

print(f'Processed {len(df_tot_m.columns)} countries')
print('Data preparation complete!')
