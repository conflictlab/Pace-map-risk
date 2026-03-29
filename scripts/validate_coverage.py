#!/usr/bin/env python3
"""
Validate that forecast CSVs include all 'active' countries from Hist.csv.
Active = sum of last 12 months in Hist.csv > 0.
Fails with exit code 1 if any active countries are missing.
"""
import sys
import pandas as pd

HIST = 'Hist.csv'
F6 = 'forecasts_h6.csv'
F12 = 'forecasts_h12.csv'

def main():
    try:
        hist = pd.read_csv(HIST, parse_dates=[0])
    except Exception as e:
        print(f"ERROR: failed to read {HIST}: {e}")
        sys.exit(2)

    # Active countries based on last 12 months
    if len(hist) < 2:
        print("WARN: Hist.csv has too few rows to validate; skipping")
        return
    tail = hist.tail(min(12, len(hist)-1))  # exclude header handled by pandas
    sums = tail.iloc[:, 1:].sum(axis=0)
    active = set(sums[sums > 0].index.tolist())

    missing = {}
    for fname in (F6, F12):
        try:
            df = pd.read_csv(fname)
        except Exception as e:
            print(f"ERROR: failed to read {fname}: {e}")
            sys.exit(2)
        cols = set(df.columns.tolist()[1:])  # drop 'date'
        miss = sorted(active - cols)
        if miss:
            missing[fname] = miss

    if missing:
        print("\n❌ Missing active countries in forecasts:")
        for k, v in missing.items():
            print(f"  {k}: {len(v)} missing → {', '.join(v[:20])}{'…' if len(v) > 20 else ''}")
        sys.exit(1)
    else:
        print("\n✅ Forecast coverage validation passed: all active countries present.")

if __name__ == '__main__':
    main()

