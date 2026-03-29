#!/usr/bin/env python3
"""
Generate best.from_site.csv (top-4 countries) from forecasts_h6.csv.
Order matches newsletter's expected CSV where the last row is the top-1 country.

Output format:
  ,name,find
  0,<4th>,0
  1,<3rd>,0
  2,<2nd>,0
  3,<1st>,0
"""
import pandas as pd

def main():
    df = pd.read_csv('forecasts_h6.csv')
    if df.columns[0].lower() != 'date':
        raise SystemExit('forecasts_h6.csv missing leading date column')
    # First data row is 1‑month ahead forecasts
    first = df.iloc[0]
    cols = df.columns.tolist()[1:]
    pairs = [(c, pd.to_numeric(first[c], errors='coerce')) for c in cols]
    pairs = [(c, float(v) if pd.notna(v) else 0.0) for c, v in pairs]
    top4 = sorted(pairs, key=lambda x: x[1], reverse=True)[:4]
    # Write rows in ascending rank order [4th, 3rd, 2nd, 1st]
    ordered = list(reversed([c for c, _ in top4]))
    out = [',name,find']
    for i, name in enumerate(ordered):
        out.append(f"{i},{name},0")
    with open('best.from_site.csv', 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('Wrote best.from_site.csv with top-4:', ', '.join(reversed(ordered)))

if __name__ == '__main__':
    main()

