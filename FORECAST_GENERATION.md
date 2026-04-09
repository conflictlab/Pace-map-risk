# PACE Forecast Generation System

## Overview

This repository generates monthly conflict fatality forecasts for both 6-month and 12-month horizons using pattern-based predictive modeling.

## Architecture

### Automated Monthly Generation

1. **GitHub Actions Workflow** (`.github/workflows/monthly-forecast-generation.yml`)
   - Runs automatically on the 1st of each month at 1:00 AM UTC
   - Can also be triggered manually via GitHub Actions UI

2. **Forecast Generation** (`generate_forecasts.py`)
   - Downloads latest UCDP data
   - Generates both h=6 and h=12 month forecasts
   - Saves outputs in standardized format
   - Creates metadata file with dates and configuration

3. **Output Preparation** (`scripts/prepare_historical_predictions.py`)
   - Organizes outputs for the Historical_Predictions directory
   - Creates properly named CSV files
   - Maintains backward compatibility

4. **Website Sync**
   - The main website repository automatically syncs from this repo
   - Sync happens daily at 3:00 AM UTC via `sync-forecasts.yml`
   - Website updates automatically after sync

## Output Files

### Forecast Files

**6-Month Forecasts:**
- `forecasts_h6.csv` - Mean forecast values
- `forecasts_h6_min.csv` - Lower bound (minimum scenario)
- `forecasts_h6_max.csv` - Upper bound (maximum scenario)

**12-Month Forecasts:**
- `forecasts_h12.csv` - Mean forecast values
- `forecasts_h12_min.csv` - Lower bound (minimum scenario)
- `forecasts_h12_max.csv` - Upper bound (maximum scenario)

**Historical Data:**
- `Hist.csv` - Full historical time series (extends as far back as UCDP data allows)

**Metadata:**
- `forecast_metadata.json` - Run information, dates, and configuration

### Backward Compatible Files (h=6 only)

For website compatibility, these files are also generated:
- `Pred_df.csv`, `Pred_df_min.csv`, `Pred_df_max.csv`
- `perc.csv`, `dec.csv`, `sta.csv`, `inc.csv`
- `saved_dictionary.pkl`, `dict_sce.pkl`, `sce_dictionary.pkl`

### Historical_Predictions Directory

Contains monthly snapshots:
- `YYYY-MM_StartDate_to_EndDate_h6.csv`
- `YYYY-MM_StartDate_to_EndDate_h12.csv`
- `latest_h6.csv`, `latest_h12.csv`
- `Hist_latest.csv`

## Manual Execution

### Prerequisites

```bash
pip install pandas numpy geopandas matplotlib seaborn shapely scipy scikit-learn pillow
pip install -r requirements.txt
```

### Run Forecast Generation

```bash
# Generate forecasts
python generate_forecasts.py

# Prepare outputs for Historical_Predictions
python scripts/prepare_historical_predictions.py
```

If the UCDP site is temporarily unreachable (timeouts), you can force offline mode
to reuse the repository's existing historical cache (`Hist.csv`/`Conf.csv`) by setting:

```bash
UCDP_OFFLINE=1 python generate_forecasts.py
```

### Backfilling Past Months (h6 + h12)

To re-generate archived months (including h12) without altering the model, run the generator as-of each month and prepare outputs. A helper script is provided:

```bash
# Examples
# 1) Specific list of months
bash scripts/backfill_months.sh --months "2024-01,2024-02,2024-03"

# 2) A range (inclusive)
bash scripts/backfill_months.sh --from 2024-01 --to 2025-12 --skip-existing-h12

# 3) Auto-detect months present without h12 and backfill them
bash scripts/backfill_months.sh --auto-missing-h12
```

Notes:
- The script uses `ASOF=YYYY-MM python generate_forecasts.py`, then runs `scripts/prepare_historical_predictions.py`.
- It commits each month and pushes via a hardened git flow (autostash rebase).
- Requires Python deps installed and `UCDP_API_TOKEN` in env if needed.
  To avoid network flakiness when backfilling via Actions, the workflow uses `UCDP_OFFLINE=1`
  so it relies on the repository's cached historical data. You can do the same locally if needed.

GitHub Actions dispatch is also available for single months: `.github/workflows/backfill-month-parallel.yml` (input `asof=YYYY-MM`).

## Configuration

### Training Window

The script uses a 10-month training window (`h_train = 10`) to align with Thomas's newsletter panels and improve scenario diversity. This is configurable in `generate_forecasts.py`.

### Forecast Horizons

Both h=6 and h=12 are generated in a single run. To modify horizons, edit the `main()` function in `generate_forecasts.py`:

```python
# Generate 6-month forecasts
results_h6 = generate_forecasts(df_tot_m, df_conf, h=6, h_train=h_train)

# Generate 12-month forecasts
results_h12 = generate_forecasts(df_tot_m, df_conf, h=12, h_train=h_train)
```

## Data Access

### For Andrea's Automated Scraping

**Recommended URLs** (always latest):
```
https://raw.githubusercontent.com/conflictlab/Pace-map-risk/main/Historical_Predictions/latest_h6.csv
https://raw.githubusercontent.com/conflictlab/Pace-map-risk/main/Historical_Predictions/latest_h12.csv
https://raw.githubusercontent.com/conflictlab/Pace-map-risk/main/Historical_Predictions/Hist_latest.csv
https://raw.githubusercontent.com/conflictlab/Pace-map-risk/main/forecast_metadata.json
```

**Direct forecast files:**
```
https://raw.githubusercontent.com/conflictlab/Pace-map-risk/main/forecasts_h6.csv
https://raw.githubusercontent.com/conflictlab/Pace-map-risk/main/forecasts_h12.csv
https://raw.githubusercontent.com/conflictlab/Pace-map-risk/main/Hist.csv
```

### Via Website (after sync)

```
https://conflictlab.github.io/data/forecasts/latest/forecasts_h6.csv
https://conflictlab.github.io/data/forecasts/latest/forecasts_h12.csv
https://conflictlab.github.io/data/forecasts/latest/Hist.csv
https://conflictlab.github.io/data/forecasts/latest/metadata.json
```

## Update Schedule

- **Forecast Generation**: 1st of each month, 1:00 AM UTC
- **Website Sync**: Daily at 3:00 AM UTC
- **Data Source**: UCDP updates their data monthly (usually mid-month)

## Troubleshooting

### GitHub Actions Failures

Check the Actions tab for detailed logs. Common issues:
- UCDP data source temporarily unavailable or timing out
- The forecast generator now performs HTTP retries/backoff for UCDP downloads
- When needed, set `UCDP_OFFLINE=1` to rely on the repo's cached `Hist.csv`/`Conf.csv`
- Insufficient historical data for pattern matching
- Memory issues (increase timeout or resources)

### Manual Testing

To test changes before monthly run:
1. Go to GitHub Actions
2. Select "Generate Monthly Forecasts" workflow
3. Click "Run workflow" → "Run workflow"

## Citation

If you use this data, please cite:

```
Schincariol, T., Chadefaux, T., et al. (2025). "Accounting for Variability in
Conflict Dynamics: A Pattern-Based Predictive Model"
```

## Contact

For issues or questions about the forecasting system, please open an issue in this repository.
