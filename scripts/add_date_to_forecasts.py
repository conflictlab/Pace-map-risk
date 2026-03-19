#!/usr/bin/env python3
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

FILES = [
    ('forecasts_h6.csv', 6),
    ('forecasts_h6_min.csv', 6),
    ('forecasts_h6_max.csv', 6),
    ('forecasts_h12.csv', 12),
    ('forecasts_h12_min.csv', 12),
    ('forecasts_h12_max.csv', 12),
    ('Pred_df.csv', 6),
    ('Pred_df_min.csv', 6),
    ('Pred_df_max.csv', 6),
]

def yyyymm_add(start: datetime, i: int) -> str:
    d = start + relativedelta(months=+i)
    return d.strftime('%Y-%m')

def main():
    with open('forecast_metadata.json','r') as f:
        meta = json.load(f)
    start_str = meta.get('forecast_start_date') or meta.get('h6_start_date') or meta.get('data_end_date')
    if not start_str:
        raise SystemExit('Missing forecast_start_date in forecast_metadata.json')
    start = datetime.strptime(start_str + '-01', '%Y-%m-%d')

    changed = []
    for fname, periods in FILES:
        try:
            df = pd.read_csv(fname)
        except Exception as e:
            print(f"Skip {fname}: {e}")
            continue
        cols = [c.lower() for c in df.columns]
        if len(df) == 0:
            print(f"Skip {fname}: empty")
            continue
        if cols and cols[0] == 'date':
            print(f"Already dated: {fname}")
            continue
        # Construct date series
        dates = [yyyymm_add(start, i) for i in range(len(df))]
        df.insert(0, 'date', dates)
        df.to_csv(fname, index=False)
        changed.append(fname)
        print(f"Updated {fname} with date column; rows={len(df)}")

    if changed:
        print("Changed files:", ', '.join(changed))
    else:
        print("No files updated.")

if __name__ == '__main__':
    main()

