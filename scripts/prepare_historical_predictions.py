#!/usr/bin/env python3
"""
Prepare forecast outputs for Historical_Predictions directory
Creates properly named CSV files for both h=6 and h=12 forecasts
"""

import pandas as pd
import json
from datetime import datetime
import os

def main():
    print("Preparing Historical_Predictions outputs...")

    # Load metadata to get dates
    with open('forecast_metadata.json', 'r') as f:
        metadata = json.load(f)

    run_date = datetime.fromisoformat(metadata['run_date'])
    forecast_month = run_date.strftime('%Y-%m')

    # Create Historical_Predictions directory if it doesn't exist
    os.makedirs('Historical_Predictions', exist_ok=True)

    # Load the forecasts
    h6 = pd.read_csv('forecasts_h6.csv', index_col=0)
    h12 = pd.read_csv('forecasts_h12.csv', index_col=0)

    # Get the forecast period ranges
    h6_start = metadata['forecast_start_date']
    h6_end = metadata['h6_end_date']
    h12_end = metadata['h12_end_date']

    # Create filenames following the existing pattern
    # Format: YYYY-MM_StartMonth-YYYY_to_EndMonth-YYYY.csv
    h6_filename = f"Historical_Predictions/{forecast_month}_{h6_start}_to_{h6_end}_h6.csv"
    h12_filename = f"Historical_Predictions/{forecast_month}_{h6_start}_to_{h12_end}_h12.csv"

    # Save the individual h6 and h12 CSVs
    h6.to_csv(h6_filename)
    h12.to_csv(h12_filename)

    print(f"✓ Created {h6_filename}")
    print(f"✓ Created {h12_filename}")

    # Create a COMBINED file for website compatibility
    # Website sync expects a single CSV with date column + country columns
    # Use h6 forecast (6-month) as the main output matching old format
    combined_filename = f"Historical_Predictions/{forecast_month}.csv"
    h6.to_csv(combined_filename)
    print(f"✓ Created {combined_filename} (combined format for website)")

    # Also create a "latest" symlink or copy for easy access
    h6.to_csv('Historical_Predictions/latest_h6.csv')
    h12.to_csv('Historical_Predictions/latest_h12.csv')
    h6.to_csv('Historical_Predictions/latest.csv')

    # Copy historical data
    hist = pd.read_csv('Hist.csv', index_col=0)
    hist.to_csv('Historical_Predictions/Hist_latest.csv')

    print(f"✓ Created latest forecast files")
    print(f"\nHistorical_Predictions directory ready for website sync")

if __name__ == '__main__':
    main()
