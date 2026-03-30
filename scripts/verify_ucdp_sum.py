#!/usr/bin/env python3
"""
Verify UCDP Candidate GED monthly sum for a country and month.

Example:
  python scripts/verify_ucdp_sum.py --year 2026 --month 01 --country "Iran"

Downloads: https://ucdp.uu.se/downloads/candidateged/GEDEvent_v{YY}_0_{M}.csv
Parses robustly, filters events where date_start.month == date_end.month == target
month and sums the 'best' field for the given country name (case-insensitive,
accepting known variants for Iran).
"""
import argparse
import io
import sys
import urllib.request
import hashlib
import pandas as pd


def fetch_candidate_csv(yy: int, m: int):
    url = f"https://ucdp.uu.se/downloads/candidateged/GEDEvent_v{yy:02d}_0_{m}.csv"
    data = urllib.request.urlopen(url, timeout=60).read()
    md5 = hashlib.md5(data).hexdigest()
    # Try a few pandas reader variants to handle delimiter quirks
    for kwargs in (
        {},
        {"sep": ";"},
        {"engine": "python"},
        {"engine": "python", "sep": None},
    ):
        try:
            df = pd.read_csv(io.BytesIO(data), **kwargs)
            return df, url, md5
        except Exception:
            continue
    raise RuntimeError(f"Unable to parse candidate CSV: {url}")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize expected columns by name (lowercase compare)
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    # Common canonical names in GED
    c_country = pick('country')
    c_start = pick('date_start')
    c_end = pick('date_end')
    c_best = pick('best')
    if not all([c_country, c_start, c_end, c_best]):
        raise RuntimeError(f"Missing expected columns; have: {list(df.columns)}")
    out = df[[c_country, c_start, c_end, c_best]].copy()
    out.columns = ['country','date_start','date_end','best']
    out['date_start'] = pd.to_datetime(out['date_start'], errors='coerce')
    out['date_end'] = pd.to_datetime(out['date_end'], errors='coerce')
    # Coerce best numeric
    out['best'] = pd.to_numeric(out['best'], errors='coerce').fillna(0)
    return out.dropna(subset=['date_start','date_end'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--month', type=int, default=1)
    ap.add_argument('--country', type=str, default='Iran')
    args = ap.parse_args()

    yy = args.year % 100
    m = args.month
    df, url, md5 = fetch_candidate_csv(yy, m)
    df = normalize_cols(df)
    print(f"Fetched: {url}\nMD5: {md5}\nRows: {len(df)}")

    # Country matching (case-insensitive).
    # UCDP may use 'Iran (Islamic Republic of)'. Accept common variants for Iran.
    wanted = {s.lower() for s in [args.country, 'Iran (Islamic Republic of)', 'IRAN (ISLAMIC REPUBLIC OF)']}
    mask_cty = df['country'].astype(str).str.lower().isin(wanted)

    # Keep events entirely in the target month (match generate_forecasts.py logic)
    ds = df.loc[mask_cty]
    same_month = (
        (ds['date_start'].dt.year == args.year) &
        (ds['date_end'].dt.year == args.year) &
        (ds['date_start'].dt.month == args.month) &
        (ds['date_end'].dt.month == args.month)
    )
    sub = ds.loc[same_month]
    total = float(sub['best'].sum())
    print(f"UCDP candidate sum for {args.country} {args.year}-{args.month:02d} (best, start=end in month): {total:.0f}")
    print(f"Events counted: {len(sub)}")
    # Debug: show top 10 events by 'best'
    if len(sub):
        top = sub.copy()
        # include id if present
        if 'id' in df.columns:
            pass
        top = top[['country','date_start','date_end','best']].sort_values('best', ascending=False).head(10)
        print("\nTop events:")
        for _, r in top.iterrows():
            print(f"  {str(r['date_start'])[:10]} — best={int(r['best'])}")
        # Group by violence type if available
        if 'type_of_violence' in df.columns:
            g = sub.groupby('type_of_violence')['best'].sum().sort_values(ascending=False)
            print("\nSum by type_of_violence:")
            print(g.to_string())

    # Also print a looser bound: any event that starts in target month/year
    looser = ds.loc[(ds['date_start'].dt.year == args.year) & (ds['date_start'].dt.month == args.month)]
    total_looser = float(looser['best'].sum())
    print(f"(Looser) sum for events starting in month: {total_looser:.0f} (events: {len(looser)})")

    # Sanity: compare to Iraq same month to rule out mis-match
    other = df[df['country'].astype(str).str.lower().eq('iraq')]
    if len(other):
        mask = (
            (other['date_start'].dt.year == args.year) &
            (other['date_end'].dt.year == args.year) &
            (other['date_start'].dt.month == args.month) &
            (other['date_end'].dt.month == args.month)
        )
        tot_other = float(other.loc[mask, 'best'].sum())
        print(f"\nIraq {args.year}-{args.month:02d} (strict) best sum: {tot_other:.0f}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)
