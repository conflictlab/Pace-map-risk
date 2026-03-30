#!/usr/bin/env python3
import os
import math
import pandas as pd
import numpy as np
try:
    import geopandas as gpd  # optional; fallback if unavailable
except Exception:  # pragma: no cover
    gpd = None
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors


RENAME = {
    'Bosnia-Herzegovina': 'Bosnia and Herz.',
    'Cambodia (Kampuchea)': 'Cambodia',
    'Central African Republic': 'Central African Rep.',
    'DR Congo (Zaire)': 'Dem. Rep. Congo',
    "Ivory Coast": "Côte d'Ivoire",
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
    'Vietnam (North Vietnam)': 'Vietnam',
}


def ensure_dirs():
    os.makedirs('Images', exist_ok=True)
    os.makedirs('docs/Images', exist_ok=True)


def load_forecasts():
    f6 = pd.read_csv('forecasts_h6.csv')
    f6_min = None
    f6_max = None
    if os.path.exists('forecasts_h6_min.csv'):
        f6_min = pd.read_csv('forecasts_h6_min.csv')
    if os.path.exists('forecasts_h6_max.csv'):
        f6_max = pd.read_csv('forecasts_h6_max.csv')
    assert f6.columns[0].lower() == 'date', 'forecasts_h6.csv missing leading date column'
    # Normalize columns
    f6 = f6.rename(columns=RENAME)
    if f6_min is not None:
        f6_min = f6_min.rename(columns=RENAME)
    if f6_max is not None:
        f6_max = f6_max.rename(columns=RENAME)
    return f6, f6_min, f6_max


def load_hist():
    # Prefer Hist.csv produced by compute; else try public one
    p = 'Hist.csv'
    if not os.path.exists(p):
        raise FileNotFoundError('Hist.csv not found — ensure compute step ran before newsletter')
    hist = pd.read_csv(p, parse_dates=[0])
    hist = hist.rename(columns={hist.columns[0]: 'date'})
    hist = hist.rename(columns=RENAME)
    return hist


def setup_fonts():
    try:
        font_path = 'Poppins/Poppins-Regular.ttf'
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'Poppins'
    except Exception:
        pass


def build_world_map(value_sum, hist_sum):
    if gpd is not None:
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            df = pd.DataFrame({'name': list(value_sum.index), 'value': list(value_sum.values)})
            dfh = pd.DataFrame({'name': list(hist_sum.index), 'hist': list(hist_sum.values)})
            w = world.merge(df, how='left', on='name').merge(dfh, how='left', on='name')
            w = w[w.name != 'Antarctica']
            w = w.fillna(0)
            w.loc[w['value'] < 0, 'value'] = 0
            w['log_per_pred'] = np.log10(w['value'] + 1)
            fig, ax = plt.subplots(1, 1, figsize=(30, 15))
            w.boundary.plot(ax=ax, color='black')
            vmax = math.ceil(float(w['log_per_pred'].max())) if len(w) else 1
            norm = mcolors.Normalize(vmin=0, vmax=vmax)
            w.plot(column='log_per_pred', cmap='Reds', ax=ax, norm=norm)
            plt.xlim(-180, 180)
            plt.box(False)
            ax.set_yticklabels([]); ax.set_yticks([]); ax.set_xticklabels([]); ax.set_xticks([])
            cbar_ax = fig.add_axes([0.65, 0.15, 0.3, 0.02])
            sm = ScalarMappable(cmap='Reds', norm=norm); sm.set_array([])
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_ticks([*range(vmax + 1)])
            cbar.set_ticklabels(['1'] + [f'$10^{e}$' for e in range(1, vmax + 1)], fontsize=20)
            plt.text(1.9, 1.5, 'Risk index', fontsize=30)
            plt.text(-8.5, 0.1, 'The risk index corresponds to the log sum of predicted fatalities in the next 6 months.', color='dimgray', fontdict={'style': 'italic', 'size': 20})
            plt.savefig('Images/map.png', bbox_inches='tight')
            plt.savefig('docs/Images/map.png', bbox_inches='tight')
            plt.close(fig)
            return
        except Exception:
            pass
    # Fallback: produce a bar chart of top 20 countries by forecast sum
    df = pd.DataFrame({'name': list(value_sum.index), 'value': list(value_sum.values)})
    top = df.sort_values('value', ascending=False).head(20).set_index('name').sort_values('value')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.barplot(x=top.index, y='value', data=top, color='red', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Top 20 predicted fatalities (6-month sum) — Fallback', fontsize=16)
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    plt.yscale('log'); ax.set_yticklabels([]); ax.set_yticks([])
    plt.savefig('Images/map.png', bbox_inches='tight')
    plt.savefig('docs/Images/map.png', bbox_inches='tight')
    plt.close(fig)


def build_global_series(hist, f6):
    # Historical: last 60 months summed across countries
    hist_series = hist.set_index('date').iloc[-60:, 1:].sum(axis=1)
    # Forecast total per month (sum across countries) for 6 months
    f6_only = f6.iloc[:, 1:]
    f6_sum = f6_only.sum(axis=1)
    # Align dates for forecast (append sequential months)
    last_date = hist_series.index[-1]
    fdates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=len(f6_sum), freq='M')
    combined = pd.concat([hist_series, pd.Series(f6_sum.values, index=fdates)])
    fig = plt.figure(figsize=(25, 6))
    d = combined.index
    plt.plot(d[:-len(f6_sum)], combined.iloc[:-len(f6_sum)], marker='o', color='grey', linestyle='-', linewidth=2, markersize=8)
    plt.plot(d[-len(f6_sum):], combined.iloc[-len(f6_sum):], marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
    plt.scatter(d[-len(f6_sum):], combined.iloc[-len(f6_sum):], color='red', s=100, zorder=5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Date', fontsize=20); plt.xticks(fontsize=16, rotation=45, ha='right'); plt.yticks(fontsize=16)
    plt.box(False)
    plt.savefig('Images/sub1_1.png', bbox_inches='tight')
    plt.savefig('docs/Images/sub1_1.png', bbox_inches='tight')
    plt.close(fig)


def barplots_by_country(value_sum, hist6):
    df = pd.DataFrame({'name': value_sum.index, 'value': value_sum.values}).merge(
        pd.DataFrame({'name': hist6.index, 'hist': hist6.values}), on='name', how='left').fillna(0)
    df['diff'] = df['value'] - df['hist']
    # sub2: Top 10 by value
    top10 = df.sort_values('value', ascending=False).head(10).set_index('name').sort_values('value')
    top10['color'] = np.where(top10['value'] > top10['hist'], 'red', 'black')
    def alpha_row(r):
        denom = r['hist'] + 1
        return np.clip(abs(r['value'] - r['hist']) / denom / 2 + 0.5, 0, 1)
    top10['alpha'] = top10.apply(alpha_row, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.barplot(x=top10.index, y='value', data=top10, palette=top10['color'], ax=ax)
    for i, bar in enumerate(ax.patches): bar.set_alpha(top10['alpha'].iloc[i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD'); ax.tick_params(bottom=False, left=False)
    plt.yscale('log'); ax.set_yticklabels([]); ax.set_yticks([])
    plt.savefig('Images/sub2.png', bbox_inches='tight'); plt.close(fig)

    # sub2_d: largest decreases (negative diff) — show absolute (positive) bars for display
    dec = df.sort_values('diff').head(10).copy()
    dec['diff_abs'] = -dec['diff']
    dec = dec.set_index('name')
    dec['color'] = np.where(dec['diff'] < 0, 'black', 'red')
    dec['alpha'] = dec.apply(alpha_row, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.barplot(x=dec.index, y='diff_abs', data=dec, palette=dec['color'], ax=ax)
    for i, bar in enumerate(ax.patches): bar.set_alpha(dec['alpha'].iloc[i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD'); ax.tick_params(bottom=False, left=False)
    plt.yscale('log'); ax.set_yticklabels([]); ax.set_yticks([])
    plt.savefig('Images/sub2_d.png', bbox_inches='tight'); plt.close(fig)

    # sub2_i: largest increases (positive diff)
    inc = df.sort_values('diff', ascending=False).head(10).copy().set_index('name')
    inc['color'] = np.where(inc['diff'] > 0, 'red', 'black')
    inc['alpha'] = inc.apply(alpha_row, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.barplot(x=inc.index, y='diff', data=inc, palette=inc['color'], ax=ax)
    for i, bar in enumerate(ax.patches): bar.set_alpha(inc['alpha'].iloc[i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD'); ax.tick_params(bottom=False, left=False)
    plt.yscale('log'); ax.set_yticklabels([]); ax.set_yticks([])
    plt.savefig('Images/sub2_i.png', bbox_inches='tight'); plt.close(fig)


def top4_details(hist, f6, f6_min, f6_max, top4):
    # For each top-4 country, produce:
    #  - exN.png: last 10 months history
    #  - exN_sce.png: 6‑month forecasts (p50) with optional band between min/max
    #  - exN_all.png: reuse exN.png for simplicity (placeholder for matches)
    hist = hist.set_index('date')
    for idx, name in enumerate(top4, start=1):
        # exN: history last 10 months
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if name in hist.columns:
            s = hist[name].tail(10)
            ax.plot(s.index, s.values, marker='o', color='black', linestyle='-', linewidth=2, markersize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        ax.set_frame_on(False)
        plt.tight_layout(); plt.savefig(f'Images/ex{idx}.png', bbox_inches='tight'); plt.close(fig)
        # exN_sce: forecasts
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # plot last 4 months history + forecast
        hx = hist[name].tail(10) if name in hist.columns else pd.Series([], dtype=float)
        ax.plot(range(len(hx)), hx.values, color='gray', linestyle='-', linewidth=2, marker='o')
        # forecast values from f6 rows 0..5
        try:
            p50 = f6[name].values
            x0 = len(hx) - 1
            xs = list(range(x0, x0 + len(p50)))
            ax.plot(xs, [hx.values[-1]] + [], color='gray', alpha=0.0)  # anchor
            # Plot forecast line continuing from last point
            base = [hx.values[-1]] if len(hx) else []
            y = pd.Series(base + list(p50)).rolling(window=1).sum().values  # simple join
            ax.plot(range(len(y))[-len(p50):], p50, color='red', linestyle='-', linewidth=2, marker='o')
            # bands
            if f6_min is not None and f6_max is not None and name in f6_min.columns and name in f6_max.columns:
                lo = f6_min[name].values; hi = f6_max[name].values
                ax.fill_between(range(len(p50)), lo, hi, color='red', alpha=0.2)
        except Exception:
            pass
        ax.set_frame_on(False); plt.tight_layout(); plt.savefig(f'Images/ex{idx}_sce.png', bbox_inches='tight'); plt.close(fig)
        # exN_all: reuse history for now
        try:
            from PIL import Image
            img = Image.open(f'Images/ex{idx}.png')
            img.save(f'Images/ex{idx}_all.png')
        except Exception:
            pass


def main():
    ensure_dirs(); setup_fonts()
    f6, f6_min, f6_max = load_forecasts()
    hist = load_hist()
    # compute values
    # value_sum: sum of forecasts across 6 months per country
    f6_only = f6.iloc[:, 1:]
    value_sum = f6_only.sum(axis=0)
    # hist6: sum of last 6 observed months per country
    hist6 = hist.set_index('date').iloc[-6:, 1:].sum(axis=0)
    # world map
    build_world_map(value_sum, hist6)
    # global series figure
    build_global_series(hist, f6)
    # bar plots
    barplots_by_country(value_sum, hist6)
    # Top-4 countries by forecasted next-6 months sum
    top4 = value_sum.sort_values(ascending=False).head(4).index.tolist()
    top4_details(hist, f6, f6_min, f6_max, top4)
    print('Newsletter images built from forecasts_h6.csv')


if __name__ == '__main__':
    main()
