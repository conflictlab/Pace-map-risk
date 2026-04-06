#!/usr/bin/env python3
"""
Generate forecasts for a chunk of countries
Usage: python generate_chunk.py <horizon> <chunk_id> <total_chunks>
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import pickle
from generate_forecasts import generate_forecasts, RENAME_MAP, rename_countries

if len(sys.argv) != 4:
    print("Usage: python generate_chunk.py <horizon> <chunk_id> <total_chunks>")
    sys.exit(1)

horizon = int(sys.argv[1])
chunk_id = int(sys.argv[2])
total_chunks = int(sys.argv[3])

print(f"Generating h={horizon} forecasts for chunk {chunk_id}/{total_chunks}")

# Load data
df_tot_m = pd.read_csv('Conf.csv', index_col=0, parse_dates=True)
df_conf = pd.read_csv('reg_coun.csv', index_col=0)

# Normalize names on both sides to avoid dropping countries like Russia
try:
    df_tot_m = rename_countries(df_tot_m)
    if df_tot_m.columns.duplicated().any():
        df_tot_m = df_tot_m.T.groupby(level=0).sum().T
    df_conf = df_conf.rename(index=RENAME_MAP)
except Exception:
    pass

common_columns = df_tot_m.columns.intersection(df_conf.index)
df_tot_m = df_tot_m.loc[:, common_columns]

# Split countries into chunks
countries = list(df_tot_m.columns)
chunk_size = len(countries) // total_chunks + (1 if len(countries) % total_chunks > chunk_id else 0)
start_idx = sum(len(countries) // total_chunks + (1 if i < len(countries) % total_chunks else 0) for i in range(chunk_id))
end_idx = start_idx + chunk_size
chunk_countries = countries[start_idx:end_idx]

print(f'Processing {len(chunk_countries)} countries')
if chunk_countries:
    print(f'First country: {chunk_countries[0]}')

# Filter to chunk
df_tot_m_chunk = df_tot_m[chunk_countries]

# Generate forecasts for this chunk
results = generate_forecasts(df_tot_m_chunk, df_conf, h=horizon, h_train=10, output_suffix=f'_h{horizon}')

# Save chunk results
results['pred_df'].to_csv(f'forecasts_h{horizon}_chunk{chunk_id}.csv')
results['pred_df_min'].to_csv(f'forecasts_h{horizon}_min_chunk{chunk_id}.csv')
results['pred_df_max'].to_csv(f'forecasts_h{horizon}_max_chunk{chunk_id}.csv')
results['df_perc'].to_csv(f'perc_h{horizon}_chunk{chunk_id}.csv')

with open(f'dict_sce_h{horizon}_chunk{chunk_id}.pkl', 'wb') as f:
    pickle.dump(results['dict_sce'], f)
with open(f'sce_dictionary_h{horizon}_chunk{chunk_id}.pkl', 'wb') as f:
    pickle.dump(results['dict_sce_plot'], f)

# Only save dict_m for h=6
if horizon == 6:
    with open(f'dict_m_h{horizon}_chunk{chunk_id}.pkl', 'wb') as f:
        pickle.dump(results['dict_m'], f)

print(f'Chunk {chunk_id} complete!')
