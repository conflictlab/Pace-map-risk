#!/usr/bin/env python3
"""
Merge parallel forecast results
Usage: python merge_results.py <horizon> <num_chunks>
"""
import pandas as pd
import pickle
import glob
import sys
import warnings

# Suppress pandas compatibility warnings
warnings.filterwarnings('ignore', category=UserWarning)

if len(sys.argv) != 3:
    print("Usage: python merge_results.py <horizon> <num_chunks>")
    sys.exit(1)

horizon = int(sys.argv[1])
num_chunks = int(sys.argv[2])

print(f'Merging h={horizon} forecasts from {num_chunks} chunks...')

# Merge CSVs
for file_pattern in [
    f'forecasts_h{horizon}_chunk',
    f'forecasts_h{horizon}_min_chunk',
    f'forecasts_h{horizon}_max_chunk',
    f'perc_h{horizon}_chunk'
]:
    chunks = []
    for i in range(num_chunks):
        f = f'h{horizon}-chunks/{file_pattern}{i}.csv'
        try:
            chunks.append(pd.read_csv(f, index_col=0))
        except FileNotFoundError:
            print(f'Warning: {f} not found')
            continue

    if chunks:
        merged = pd.concat(chunks, axis=1)
        output_name = file_pattern.replace('_chunk', '').replace(f'perc_h{horizon}', 'perc' if horizon == 6 else f'perc_h{horizon}') + '.csv'
        merged.to_csv(output_name)
        print(f'Saved {output_name}')

# Merge pickles
pickle_patterns = [
    (f'dict_sce_h{horizon}_chunk', f'dict_sce{"" if horizon == 6 else "_h" + str(horizon)}.pkl'),
    (f'sce_dictionary_h{horizon}_chunk', f'sce_dictionary{"" if horizon == 6 else "_h" + str(horizon)}.pkl')
]

# Add dict_m only for h=6
if horizon == 6:
    pickle_patterns.insert(0, (f'dict_m_h{horizon}_chunk', 'saved_dictionary.pkl'))

for file_pattern, output_name in pickle_patterns:
    merged_dict = {}
    for i in range(num_chunks):
        f = f'h{horizon}-chunks/{file_pattern}{i}.pkl'
        try:
            with open(f, 'rb') as pf:
                # Use pandas compat mode to handle version differences
                chunk_dict = pickle.load(pf)
                merged_dict.update(chunk_dict)
        except FileNotFoundError:
            print(f'Warning: {f} not found')
            continue
        except (ModuleNotFoundError, AttributeError) as e:
            # Handle pandas version incompatibility
            print(f'Warning: Could not load {f} due to pandas version incompatibility: {e}')
            print(f'Skipping {f} - you may need to regenerate this chunk')
            continue

    if merged_dict:
        with open(output_name, 'wb') as pf:
            pickle.dump(merged_dict, pf, protocol=4)  # Use protocol 4 for better compatibility
        print(f'Saved {output_name}')
    else:
        print(f'Warning: No data to save for {output_name}')

# Create backward-compatible files for h=6
if horizon == 6:
    pd.read_csv(f'forecasts_h{horizon}.csv', index_col=0).to_csv('Pred_df.csv')
    pd.read_csv(f'forecasts_h{horizon}_min.csv', index_col=0).to_csv('Pred_df_min.csv')
    pd.read_csv(f'forecasts_h{horizon}_max.csv', index_col=0).to_csv('Pred_df_max.csv')
    print('Created backward-compatible files')

print(f'h={horizon} merge complete!')

# Post-process: rescale normalized forecasts to historical units
# We apply scaling per country using the last 24 months of data from Conf.csv
try:
    import os
    import pandas as pd
    conf_path = os.path.join('data', 'Conf.csv')
    if os.path.exists(conf_path):
        conf = pd.read_csv(conf_path, index_col=0, parse_dates=True)

        def _rescale_file(path):
            if not os.path.exists(path):
                return
            df = pd.read_csv(path, index_col=0)
            # Ensure we only scale columns present in Conf.csv
            common_cols = [c for c in df.columns if c in conf.columns]
            if not common_cols:
                return
            scaled = df.copy()
            for col in common_cols:
                # Use last 24 months where available; fall back to all if shorter
                series = conf[col].dropna()
                window = series.iloc[-24:] if len(series) >= 24 else series
                a = window.max() - window.min()
                b = window.min()
                # Apply affine rescaling
                scaled[col] = df[col] * a + b
            # Clamp negatives to zero
            scaled[scaled < 0] = 0
            scaled.to_csv(path)

        # Apply to merged outputs for this horizon
        _rescale_file(f'forecasts_h{horizon}.csv')
        _rescale_file(f'forecasts_h{horizon}_min.csv')
        _rescale_file(f'forecasts_h{horizon}_max.csv')
        print(f'Rescaled forecasts_h{horizon}*.csv to historical units')
    else:
        print('Warning: data/Conf.csv not found; skipping rescaling')
except Exception as e:
    print(f'Warning: Failed to rescale forecasts: {e}')
