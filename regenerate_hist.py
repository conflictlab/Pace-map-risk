#!/usr/bin/env python3
"""
Regenerate Hist.csv from Conf.csv with correct country name mappings
"""
import pandas as pd

def rename_countries(df):
    """Apply standard country name mappings"""
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

# Read Conf.csv
print("Reading Conf.csv...")
df = pd.read_csv('Conf.csv', index_col=0, parse_dates=True)
print(f"  Loaded {len(df)} months of data")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")

# Apply country name mappings
print("\nApplying country name mappings...")
df_renamed = rename_countries(df)

# Save as Hist.csv
print("\nSaving to Hist.csv...")
df_renamed.to_csv('Hist.csv')

print(f"\n✓ SUCCESS: Hist.csv regenerated with {len(df_renamed)} months of data")
print(f"  Start date: {df_renamed.index[0].strftime('%Y-%m')}")
print(f"  End date: {df_renamed.index[-1].strftime('%Y-%m')}")
print(f"  Countries: {len(df_renamed.columns)}")
