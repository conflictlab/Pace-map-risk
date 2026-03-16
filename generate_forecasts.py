# -*- coding: utf-8 -*-
"""
PACE Forecast Generation Script
Generates both 6-month and 12-month forecasts
Modified to support multiple forecast horizons and extended historical data
"""

import pandas as pd
from shape import Shape, finder
import numpy as np
import pickle
from datetime import datetime, timedelta
import json
import os

def rename_countries(df):
    """Apply standard country name mappings"""
    return df.rename(columns={
        'Bosnia-Herzegovina': 'Bosnia and Herz.',
        'Cambodia (Kampuchea)': 'Cambodia',
        'Central African Republic': 'Central African Rep.',
        'DR Congo (Zaire)': 'Dem. Rep. Congo',
        'Ivory Coast': 'Côte d\'Ivoire',
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

def generate_forecasts(df_tot_m, df_conf, h=6, h_train=10, output_suffix=''):
    """
    Generate forecasts for a given horizon h

    Parameters:
    -----------
    df_tot_m : pd.DataFrame
        Monthly aggregated fatality data
    df_conf : pd.Series
        Country configuration mapping
    h : int
        Forecast horizon in months (default: 6)
    h_train : int
        Training window size in months (default: 10)
    output_suffix : str
        Suffix for output files (e.g., '_h6', '_h12')

    Returns:
    --------
    dict : Dictionary containing all forecast outputs
    """
    print(f"Generating {h}-month ahead forecasts...")

    # Initialize data structures
    df_next = {
        0: pd.DataFrame(columns=df_tot_m.columns, index=range(h_train+h)),
        1: pd.DataFrame(columns=df_tot_m.columns, index=range(h_train+h)),
        2: pd.DataFrame(columns=df_tot_m.columns, index=range(h_train+h))
    }
    df_perc = pd.DataFrame(columns=df_tot_m.columns, index=range(3))
    dict_sce = {i: [[], [], []] for i in df_tot_m.columns}

    pred_tot = []
    pred_raw = []
    dict_m = {i: [] for i in df_tot_m.columns}
    dict_sce_plot = {i: [[], []] for i in df_tot_m.columns}

    # Generate forecasts for each country
    for coun in range(len(df_tot_m.columns)):
        if not (df_tot_m.iloc[-h_train:, coun] == 0).all():
            shape = Shape()
            shape.set_shape(df_tot_m.iloc[-h_train:, coun])
            find = finder(df_tot_m.iloc[:-h, :], shape)
            find.find_patterns(min_d=0.1, select=True, metric='dtw',
                             dtw_sel=2, min_mat=3, d_increase=0.05)

            pred_ori = find.predict(horizon=h, plot=False, mode='mean')
            find.create_sce(df_conf, h)
            pred_raw.append(pred_ori)
            sce_ts = find.val_sce
            sce_ts.columns = pd.date_range(
                start=df_tot_m.iloc[-h_train:, coun].index[-1] + pd.DateOffset(months=1),
                periods=h,
                freq='M'
            )
            dict_sce_plot[df_tot_m.columns[coun]][0] = find.sce
            dict_sce_plot[df_tot_m.columns[coun]][1] = sce_ts

            pred_ori = pred_ori * (df_tot_m.iloc[-h_train:, coun].max() -
                                  df_tot_m.iloc[-h_train:, coun].min()) + \
                       df_tot_m.iloc[-h_train:, coun].min()
            pred_tot.append(pred_ori)
            dict_m[df_tot_m.columns[coun]] = find.sequences
            seq_pred = find.predict(horizon=h, plot=False, mode='mean', seq_out=True)
            y_pred = np.full((len(seq_pred),), np.nan)
            y_pred[seq_pred.sum(axis=1) <= 1.5] = 0
            y_pred[(seq_pred.sum(axis=1) > 1.5) & (seq_pred.sum(axis=1) < 5)] = 1
            y_pred[seq_pred.sum(axis=1) >= 5] = 2

            perc = []
            for i in range(3):
                if len(seq_pred[y_pred == i]) != 0:
                    norm = seq_pred[y_pred == i].mean() * \
                           (df_tot_m.iloc[-h_train:, coun].max() -
                            df_tot_m.iloc[-h_train:, coun].min()) + \
                           df_tot_m.iloc[-h_train:, coun].min()
                    norm.index = pd.date_range(
                        start=df_tot_m.iloc[-h_train:, coun].index[-1] + pd.DateOffset(months=1),
                        periods=h,
                        freq='M'
                    )
                    seq_f = pd.concat([df_tot_m.iloc[-h_train:, coun], norm], axis=0)
                    index_s = seq_f.index
                    seq_f = seq_f.reset_index(drop=True)
                    df_next[i].iloc[:, coun] = seq_f
                else:
                    df_next[i].iloc[:, coun] = [float('nan')] * (h_train + h)
                perc.append(round(len(seq_pred[y_pred == i]) / len(seq_pred) * 100, 1))
            df_perc.iloc[:, coun] = perc

            y_pred_p = y_pred.astype(float)
            count_zeros = 0
            count_ones = 0
            count_twos = 0
            for i in range(len(y_pred_p)):
                value = y_pred_p[i]
                norm = seq_pred.iloc[i, :] * (find.sequences[i][0].max() -
                                              find.sequences[i][0].min()) + \
                       find.sequences[i][0].min()
                norm.index = pd.date_range(
                    start=find.sequences[i][0].index[-1] + pd.DateOffset(months=1),
                    periods=h,
                    freq='M'
                )
                seq_f = pd.concat([find.sequences[i][0], norm], axis=0)
                seq_f.name = find.sequences[i][0].name
                if value == 0 and count_zeros < 5:
                    dict_sce[df_tot_m.columns[coun]][0].append(seq_f)
                    count_zeros += 1
                elif value == 1 and count_ones < 5:
                    dict_sce[df_tot_m.columns[coun]][1].append(seq_f)
                    count_ones += 1
                elif value == 2 and count_twos < 5:
                    dict_sce[df_tot_m.columns[coun]][2].append(seq_f)
                    count_twos += 1
        else:
            pred_tot.append(pd.DataFrame(np.zeros((h, 3))))
            pred_raw.append(pd.DataFrame(np.zeros((h, 3))))

    # Process and save predictions (mean) - SCALE to historical units
    pred_df = [df.iloc[:, 0] for df in pred_tot]
    pred_df = pd.concat(pred_df, axis=1)
    pred_df.columns = df_tot_m.columns
    pred_df = rename_countries(pred_df)

    # Process and save predictions (min) - SCALE to historical units
    pred_df_min = [df.iloc[:, 1] for df in pred_tot]
    pred_df_min = pd.concat(pred_df_min, axis=1)
    pred_df_min.columns = df_tot_m.columns
    pred_df_min = rename_countries(pred_df_min)

    # Process and save predictions (max) - SCALE to historical units
    pred_df_max = [df.iloc[:, 2] for df in pred_tot]
    pred_df_max = pd.concat(pred_df_max, axis=1)
    pred_df_max.columns = df_tot_m.columns
    pred_df_max = rename_countries(pred_df_max)

    # Apply country renaming to other dataframes
    df_perc = rename_countries(df_perc)
    df_next[0] = rename_countries(df_next[0])
    df_next[1] = rename_countries(df_next[1])
    df_next[2] = rename_countries(df_next[2])

    rena = {
        'Bosnia-Herzegovina': 'Bosnia and Herz.',
        'Cambodia (Kampuchea)': 'Cambodia',
        'Central African Republic': 'Central African Rep.',
        'DR Congo (Zaire)': 'Dem. Rep. Congo',
        'Ivory Coast': 'Côte d\'Ivoire',
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
    }
    dict_sce_f = {rena[key] if key in rena else key: item for key, item in dict_sce.items()}
    dict_sce_plot_f = {rena[key] if key in rena else key: item for key, item in dict_sce_plot.items()}

    return {
        'pred_df': pred_df,
        'pred_df_min': pred_df_min,
        'pred_df_max': pred_df_max,
        'df_perc': df_perc,
        'df_next': df_next,
        'dict_m': dict_m,
        'dict_sce': dict_sce_f,
        'dict_sce_plot': dict_sce_plot_f,
        'pred_tot': pred_tot,
        'pred_raw': pred_raw
    }

def main():
    """Main execution function"""
    print("=" * 60)
    print("PACE Forecast Generation")
    print("=" * 60)

    # Load and process UCDP data
    print("\n1. Loading UCDP data...")
    df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged251-csv.zip",
                     parse_dates=['date_start', 'date_end'], low_memory=False)
    month = datetime.now().strftime("%m")

    if month == '01':
        month = '13'
    # Use Candidate GED monthly CSVs matching the current GED major stream (e.g., 2026 -> v26_0_{i}.csv)
    major = int(datetime.now().strftime('%y'))  # 2026 -> 26
    def _read_candidate_csv(url: str) -> pd.DataFrame:
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
        try:
            df_can = _read_candidate_csv(f'https://ucdp.uu.se/downloads/candidateged/GEDEvent_v{major}_0_{i}.csv')
            df_can.columns = df.columns
            df_can['date_start'] = pd.to_datetime(df_can['date_start'])
            df_can['date_end'] = pd.to_datetime(df_can['date_end'])
            df_can = df_can.drop_duplicates()
            df = pd.concat([df, df_can], axis=0)
        except:
            print(f"   Note: Could not load candidate data for v{major}_0_{i}.csv")

    # Backfill previous GED stream for the full prior year (e.g., 2025 -> v25_0_{1..12}.csv)
    prev_major = major - 1
    for i in range(1, 13):
        try:
            df_can = _read_candidate_csv(f'https://ucdp.uu.se/downloads/candidateged/GEDEvent_v{prev_major}_0_{i}.csv')
            df_can.columns = df.columns
            df_can['date_start'] = pd.to_datetime(df_can['date_start'])
            df_can['date_end'] = pd.to_datetime(df_can['date_end'])
            df_can = df_can.drop_duplicates()
            df = pd.concat([df, df_can], axis=0)
        except:
            print(f"   Note: Could not load candidate data for v{prev_major}_0_{i}.csv")

    print("2. Processing data...")
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
    del df
    del df_tot

    df_conf = pd.read_csv('reg_coun.csv', index_col=0, squeeze=True)
    common_columns = df_tot_m.columns.intersection(df_conf.index)
    df_tot_m = df_tot_m.loc[:, common_columns]

    # Save full historical data
    print("3. Saving historical data...")
    hist_full = rename_countries(df_tot_m)
    hist_full.to_csv('Hist.csv')
    print(f"   Saved Hist.csv with {len(hist_full)} months of data")

    # Use extended training window (24 months instead of 10)
    h_train = 24

    # Generate 6-month forecasts
    print("\n4. Generating 6-month forecasts...")
    results_h6 = generate_forecasts(df_tot_m, df_conf, h=6, h_train=h_train, output_suffix='_h6')

    # Save 6-month outputs
    results_h6['pred_df'].to_csv('forecasts_h6.csv')
    results_h6['pred_df_min'].to_csv('forecasts_h6_min.csv')
    results_h6['pred_df_max'].to_csv('forecasts_h6_max.csv')
    results_h6['df_perc'].to_csv('perc_h6.csv')

    # Save backward compatible filenames for h=6
    results_h6['pred_df'].to_csv('Pred_df.csv')
    results_h6['pred_df_min'].to_csv('Pred_df_min.csv')
    results_h6['pred_df_max'].to_csv('Pred_df_max.csv')
    results_h6['df_perc'].to_csv('perc.csv')
    results_h6['df_next'][0].to_csv('dec.csv')
    results_h6['df_next'][1].to_csv('sta.csv')
    results_h6['df_next'][2].to_csv('inc.csv')

    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(results_h6['dict_m'], f)
    with open('dict_sce.pkl', 'wb') as f:
        pickle.dump(results_h6['dict_sce'], f)
    with open('sce_dictionary.pkl', 'wb') as f:
        pickle.dump(results_h6['dict_sce_plot'], f)

    print("   ✓ Saved 6-month forecasts")

    # Generate 12-month forecasts
    print("\n5. Generating 12-month forecasts...")
    results_h12 = generate_forecasts(df_tot_m, df_conf, h=12, h_train=h_train, output_suffix='_h12')

    # Save 12-month outputs
    results_h12['pred_df'].to_csv('forecasts_h12.csv')
    results_h12['pred_df_min'].to_csv('forecasts_h12_min.csv')
    results_h12['pred_df_max'].to_csv('forecasts_h12_max.csv')
    results_h12['df_perc'].to_csv('perc_h12.csv')
    results_h12['df_next'][0].to_csv('dec_h12.csv')
    results_h12['df_next'][1].to_csv('sta_h12.csv')
    results_h12['df_next'][2].to_csv('inc_h12.csv')

    with open('dict_sce_h12.pkl', 'wb') as f:
        pickle.dump(results_h12['dict_sce'], f)
    with open('sce_dictionary_h12.pkl', 'wb') as f:
        pickle.dump(results_h12['dict_sce_plot'], f)

    print("   ✓ Saved 12-month forecasts")

    # Create metadata file
    print("\n6. Creating metadata...")
    metadata = {
        'run_date': datetime.now().isoformat(),
        'data_end_date': last_month.strftime('%Y-%m'),
        'forecast_start_date': (last_month + pd.DateOffset(months=1)).strftime('%Y-%m'),
        'h6_end_date': (last_month + pd.DateOffset(months=6)).strftime('%Y-%m'),
        'h12_end_date': (last_month + pd.DateOffset(months=12)).strftime('%Y-%m'),
        'training_window_months': h_train,
        'historical_start_date': df_tot_m.index[0].strftime('%Y-%m'),
        'historical_end_date': last_month.strftime('%Y-%m'),
        'total_historical_months': len(df_tot_m)
    }

    with open('forecast_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("   ✓ Saved metadata")

    print("\n" + "=" * 60)
    print("✓ Forecast generation complete!")
    print("=" * 60)
    print("\nOutputs:")
    print("  - Hist.csv (full historical data)")
    print("  - forecasts_h6.csv, forecasts_h6_min.csv, forecasts_h6_max.csv")
    print("  - forecasts_h12.csv, forecasts_h12_min.csv, forecasts_h12_max.csv")
    print("  - forecast_metadata.json")
    print("\nBackward compatible outputs (h=6):")
    print("  - Pred_df.csv, Pred_df_min.csv, Pred_df_max.csv")
    print("  - perc.csv, dec.csv, sta.csv, inc.csv")

if __name__ == '__main__':
    main()
