#!/usr/bin/env python3
"""
Prepare archived forecast CSVs under Historical_Predictions/ for each run.

Creates (if inputs exist):
  - Historical_Predictions/YYYY-MM_h6.csv
  - Historical_Predictions/YYYY-MM_h12.csv
  - Historical_Predictions/latest_h6.csv (copy of forecasts_h6.csv)
  - Historical_Predictions/latest_h12.csv (copy of forecasts_h12.csv)
  - Historical_Predictions/Hist_latest.csv (copy of Hist.csv)

Derives YYYY-MM from forecast_metadata.json (forecast_start_date), falling back to data_end_date.
Does not recompute forecasts — just copies/archives the files produced by the workflow.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    meta_path = ROOT / 'forecast_metadata.json'
    if not meta_path.exists():
        print('forecast_metadata.json not found; skipping archive prep')
        return
    meta = json.loads(meta_path.read_text())
    period = meta.get('forecast_start_date') or meta.get('data_end_date')
    if not period:
        print('Missing forecast_start_date/data_end_date in metadata; skipping')
        return

    out_dir = ROOT / 'Historical_Predictions'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map of input files to archive names
    pairs = [
        (ROOT / 'forecasts_h6.csv', out_dir / f'{period}_h6.csv'),
        (ROOT / 'forecasts_h12.csv', out_dir / f'{period}_h12.csv'),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copyfile(src, dst)
            print(f'Wrote {dst}')
        else:
            print(f'Skip missing {src.name}')

    # Also maintain latest copies for convenience
    latest_pairs = [
        (ROOT / 'forecasts_h6.csv', out_dir / 'latest_h6.csv'),
        (ROOT / 'forecasts_h12.csv', out_dir / 'latest_h12.csv'),
        (ROOT / 'Hist.csv', out_dir / 'Hist_latest.csv'),
    ]
    for src, dst in latest_pairs:
        if src.exists():
            shutil.copyfile(src, dst)
            print(f'Wrote {dst}')

if __name__ == '__main__':
    main()

