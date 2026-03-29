# Pace Map Risk

This repository provides live conflict risk predictions across countries based on the UCDP conflict dataset. It computes country-level fatality forecasts using shape pattern recognition and generates reports including visual maps, historical trends, and risk rankings. The predictions aim to identify potential conflict escalation over a 6-month horizon.

## 💡 Main Features

- Live monthly conflict forecast  
- Pattern-based similarity modeling for country-level predictions  
- Scenario-based risk at the country level  
- 6-month horizon, monthly updated live forecast
- Map and newsletter generation for distribution-ready summaries

---

## 🚀 Scripts Overview

### `test.py` — **Live Prediction Engine**

This is the main execution script. It performs:
- Download and aggregation of UCDP and candidate GED datasets
- Conflict fatality aggregation by country and month
- Time-series shape matching for each country (last 10 months)
- 6-month ahead predictions
- Generation of output datasets and maps

### `crea_pred_csv.py` — **CSV Generator**

Converts forecast results into clean CSV tables, suitable for analysis or dashboarding.

### `newsletter.py` — **Newsletter Assembly**

Formats results into a structured text or markdown-based newsletter overview.

### `pdf.py` — **PDF Renderer**

Converts the newsletter or other outputs into PDF format using charts and country highlights.

---

## Outputs of the Model

### ✅ `Historical_Predictions/*.csv` — **Live Forecast CSV**
- **Type:** CSV.
- **Dimensions:** `h × N` (where `h = 6` months, and `N` is the country name).
- **Content:** Predicted conflict fatalities for each country at each future time step.
- **Index:** Future time steps (t+1 to t+6).
- **Columns:** Country Names
- **Purpose:** Main deliverable; this file represents live forecast.
- **Name Format** The format is the following: Year(input)-Month(input)_Month-Year(first forecasted month)_Month-Year(last forecasted month). For example, 2024-01_Feb-2024_to_Jul-2024.csv contains the forecasted value from February 2024 to July 2024 using data until January 2024.

---

### ✅ `sce_dictionary.pkl` — **Scenarios pre country**
- **Type:** Pickle (list of lists).
- **Content:** For each country index, the list of scenarios with their associated probabilities.

---

### 🧩 `saved_dictionary.pkl` — **Matched Subpatterns**
- **Type:** Pickle (dictionary).
- **Content:** For each country index,  the similar sequences in the historical dataset.
- **Purpose:** Diagnostic and interpretability — reveals which historical patterns contributed to each forecast.

---

**Additional Files for Report**

| File | Description |
|------|-------------|
| `Pred_df.csv` | Raw 6-month prediction (average fatalities) per country |
| `Pred_df_min.csv` | Lower-bound prediction (minimum fatalities) |
| `Pred_df_max.csv` | Upper-bound prediction (maximum fatalities) |
| `dec.csv` | Predicted trajectories for countries expected to **decrease** in conflict |
| `sta.csv` | Predicted trajectories for countries expected to **remain stable** |
| `inc.csv` | Predicted trajectories for countries expected to **increase** in conflict |
| `perc.csv` | Percentage of scenarios in each risk class (dec/stable/inc) per country |
| `best.csv` | Top 4 countries with highest projected risk |
| `dict_sce.pkl` | Simple Scenario-based sequences for top countries (structured data) (up-still-down) |
| `world_plot.geojson` | GeoJSON world map with risk and population-adjusted predictions |
| `Images/*.png` | Visual outputs: global risk map, charts, country-specific prediction trajectories |

---

## Keep Newsletter Aligned With Website

To ensure the PDF newsletter highlights the same top-risk countries as the website dashboard:

- Generate a site-derived top-4 list (from `content/forecasts/latest.json`):
  - `node scripts/generate-newsletter-best.js` (writes `Pace-map-risk/best.from_site.csv`)
- The PDF generator (`Pace-map-risk/pdf.py`) will prefer `best.from_site.csv` when present; otherwise it falls back to `best.csv`.

This keeps the newsletter’s country selection consistent with the site’s latest published snapshot.
