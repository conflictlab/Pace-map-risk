#!/usr/bin/env python3
"""
Generate newsletter panels using ThomasSchinca's layout directly from existing
pickles and Hist.csv, aligned with best.from_site.csv selection.

Outputs (in ./Images):
  - ex1.png, ex2.png, ex3.png, ex4.png            (full history plots)
  - ex1_all.png, ex2_all.png, ex3_all.png, ex4_all.png  (2x2 closest matches)
  - ex1_sce.png, ex2_sce.png, ex3_sce.png, ex4_sce.png  (scenario mosaics)
"""
import os
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, FuncFormatter

SAVE_KW = dict(bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=220)

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
RENAME_REV = {v: k for k, v in RENAME.items()}


def ensure_dirs():
    os.makedirs('Images', exist_ok=True)


def load_hist():
    hist = pd.read_csv('Hist.csv', parse_dates=[0])
    hist = hist.rename(columns={hist.columns[0]: 'date'})
    hist = hist.rename(columns=RENAME)
    return hist


def load_pickles():
    with open('saved_dictionary.pkl', 'rb') as f:
        dict_m = pickle.load(f)
    with open('sce_dictionary.pkl', 'rb') as f:
        dict_sce_plot_f = pickle.load(f)
    # Ensure renamed keys exist
    dict_m_ren = {RENAME.get(k, k): v for k, v in dict_m.items()}
    dict_sce_plot_ren = {RENAME.get(k, k): v for k, v in dict_sce_plot_f.items()}
    return dict_m_ren, dict_sce_plot_ren


def resolve_top4():
    # Prefer site-derived list
    for path in ('best.from_site.csv', 'best.csv'):
        if os.path.exists(path):
            dfb = pd.read_csv(path)
            if 'name' in dfb.columns:
                names = dfb['name'].tolist()
                # Files are ordered [4th,3rd,2nd,1st]; return [1st,2nd,3rd,4th]
                names = list(reversed(names))
                return names[:4]
    # Fallback: first-row forecasts
    f6 = pd.read_csv('forecasts_h6.csv')
    first = f6.iloc[0]
    cols = f6.columns.tolist()[1:]
    pairs = [(c, pd.to_numeric(first[c], errors='coerce')) for c in cols]
    pairs = [(c, float(v) if pd.notna(v) else 0.0) for c, v in pairs]
    return [c for c, _ in sorted(pairs, key=lambda x: x[1], reverse=True)[:4]]


def plot_history(hist, name, idx):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    if name in hist.columns:
        s = hist.set_index('date')[name]
        ax.plot(s.index, s.values, marker='o', color='black', linestyle='-', linewidth=3, markersize=6)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_frame_on(False)
    plt.tight_layout(); plt.savefig(f'Images/ex{idx}.png', **SAVE_KW); plt.close(fig)


def plot_matches(dict_m, name, idx):
    # Thomas uses original name for dict_m access
    name_before = RENAME_REV.get(name, name)
    panel = []
    if name_before in dict_m:
        for c in range(4):
            try:
                series = dict_m[name_before][c][0]
                dist = dict_m[name_before][c][1] if len(dict_m[name_before][c]) > 1 else 0.0
                panel.append((dist, series))
            except Exception:
                break
    if not panel:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    for j, (dist, series) in enumerate(panel[:4]):
        axp = axes[j]
        try:
            axp.set_facecolor('white')
            axp.plot(series.index, series.values, color='#808080', linestyle='-', linewidth=2, marker='o', markersize=4)
            axp.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
            axp.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=4))
            axp.set_title(series.name, fontsize=12, color='#808080')
        except Exception:
            axp.set_facecolor('white')
            axp.plot(range(len(series)), list(series), color='#808080', linestyle='-', linewidth=2, marker='o', markersize=4)
            axp.set_title(f'Match {j+1}', fontsize=12, color='#808080')
        axp.set_frame_on(False)
        axp.set_xticks([]); axp.set_yticks([])
    for j in range(len(panel), 4):
        axes[j].axis('off')
    plt.tight_layout(); plt.savefig(f'Images/ex{idx}_all.png', **SAVE_KW); plt.close(fig)


def plot_scenarios(hist, dict_sce_plot, name, idx):
    if name not in dict_sce_plot:
        return
    try:
        scen_df = dict_sce_plot[name][1]
    except Exception:
        return
    if scen_df is None or len(scen_df) == 0:
        return
    num = len(scen_df)
    if num > 2:
        layout = [[0,0,0,0,0,3], [1,1,1,1,1,4], [2,2,2,2,2,5]]
    elif num == 2:
        layout = [[2,2,2,2,2,9],[0,0,0,0,0,3],[0,0,0,0,0,3],[0,0,0,0,0,3],[5,5,5,5,5,6],[1,1,1,1,1,4],[1,1,1,1,1,4],[1,1,1,1,1,4],[7,7,7,7,7,8]]
    else:
        layout = [[1,1,1,1,1,4], [0,0,0,0,0,3], [2,2,2,2,2,5]]
    fig, ax = plt.subplot_mosaic(layout, figsize=(10, 8))
    fig.patch.set_facecolor('white')
    base = hist.set_index('date')[name] if name in hist.columns else pd.Series([], dtype=float)
    if len(base) and (base.max() - base.min()) != 0:
        b = (base - base.min()) / (base.max() - base.min())
    else:
        b = base.copy()
    # Select up to 3 scenario rows by probability (index)
    try:
        order_idx = pd.Series(scen_df.index).sort_values(ascending=False).index[:3]
        sel = scen_df.iloc[order_idx, :]
    except Exception:
        sel = scen_df.iloc[:3, :]
    for c, (p, row) in enumerate(sel.iterrows()):
        prob = float(p) if not isinstance(p, tuple) else float(p[0])
        if prob >= 0.5:
            color = "#df2226"
        else:
            sup1 = f"{34 + int((0.5 - prob)*100*3):x}"
            sup2 = f"{38 + int((0.5 - prob)*100*3):x}"
            color = f"#df{sup1}{sup2}"
        scen = pd.Series(b.tolist() + row.tolist())
        if len(base) and (base.max() - base.min()) != 0:
            scen = scen * (base.max() - base.min()) + base.min()
        ax_line = ax[c]
        ax_text = ax.get(c+3, None)
        ax_line.set_facecolor('white')
        ax_line.plot(scen, color='gray', linestyle='-', linewidth=2)
        ax_line.plot(scen.iloc[-7:], color=color, linestyle='-', linewidth=5)
        ax_line.set_frame_on(False)
        ax_line.set_xticks([10,11,12,13,14,15], [f't+{i}' if i not in [1,3,5] else '' for i in range(1,7)])
        ax_line.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax_line.tick_params(axis='y', labelsize=20)
        ax_line.tick_params(axis='x', labelsize=20, rotation=30)
        ax_line.grid('y', alpha=0.5)
        for sp in ['top','right','bottom','left']:
            ax_line.spines[sp].set_visible(False)
        if ax_text is not None:
            ax_text.text(0.1, 0.4, f'Freq = {int(prob*100)}%', fontsize=30, color=color)
            ax_text.set_frame_on(False)
            ax_text.set_xticks([]); ax_text.set_yticks([])
    plt.tight_layout(); plt.savefig(f'Images/ex{idx}_sce.png', **SAVE_KW); plt.close(fig)


def main():
    ensure_dirs()
    hist = load_hist()
    dict_m, dict_sce_plot = load_pickles()
    top4 = resolve_top4()
    for i, name in enumerate(top4, start=1):
        plot_history(hist, name, i)
        plot_matches(dict_m, name, i)
        plot_scenarios(hist, dict_sce_plot, name, i)
    print('✓ Generated Thomas-style ex panels for:', ', '.join(top4))


if __name__ == '__main__':
    main()

