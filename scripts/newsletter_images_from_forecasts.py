#!/usr/bin/env python3
import os
import math
import pandas as pd
import numpy as np
import pickle
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
from typing import Optional, List, Tuple

try:
    # Optional; only used for on-the-fly fallback computation of matches/scenarios
    from shape import Shape, finder as _Finder
except Exception:  # pragma: no cover
    Shape = None
    _Finder = None


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
    os.makedirs('docs/Images', exist_ok=True)

# High-DPI, opaque white saves so small panels remain readable in PDF
SAVE_KW = dict(bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=220)


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


def load_pickles():
    sce = None
    matches = None
    for fname, var in (( 'sce_dictionary.pkl', 'sce'), ('saved_dictionary.pkl','matches')):
        if os.path.exists(fname):
            try:
                with open(fname, 'rb') as f:
                    obj = pickle.load(f)
                if fname.startswith('sce_'):
                    sce = obj
                else:
                    matches = obj
            except Exception:
                pass
    return sce, matches


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
            fig.patch.set_facecolor('white'); ax.set_facecolor('white')
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
            plt.savefig('Images/map.png', **SAVE_KW)
            plt.savefig('docs/Images/map.png', **SAVE_KW)
            plt.close(fig)
            return
        except Exception:
            pass
    # Fallback: produce a bar chart of top 20 countries by forecast sum
    df = pd.DataFrame({'name': list(value_sum.index), 'value': list(value_sum.values)})
    top = df.sort_values('value', ascending=False).head(20).set_index('name').sort_values('value')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    sns.barplot(x=top.index, y='value', data=top, color='red', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Top 20 predicted fatalities (6-month sum) — Fallback', fontsize=16)
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    plt.yscale('log'); ax.set_yticklabels([]); ax.set_yticks([])
    plt.savefig('Images/map.png', **SAVE_KW)
    plt.savefig('docs/Images/map.png', **SAVE_KW)
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
    fig.patch.set_facecolor('white')
    d = combined.index
    # Historical (black)
    plt.plot(
        d[:-len(f6_sum)], combined.iloc[:-len(f6_sum)],
        marker='o', color='black', linestyle='-', linewidth=2, markersize=8
    )
    # Predicted (red)
    plt.plot(
        d[-len(f6_sum):], combined.iloc[-len(f6_sum):],
        marker='o', color='red', linestyle='-', linewidth=2, markersize=8
    )
    plt.scatter(d[-len(f6_sum):], combined.iloc[-len(f6_sum):], color='red', s=100, zorder=5)
    # Connect last historical point to first forecast point with a black line
    try:
        last_hist_date = hist_series.index[-1]
        first_fc_date = fdates[0]
        last_hist_val = float(hist_series.iloc[-1])
        first_fc_val = float(f6_sum.iloc[0])
        plt.plot([last_hist_date, first_fc_date], [last_hist_val, first_fc_val], color='black', linewidth=2)
    except Exception:
        pass
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Date', fontsize=20); plt.xticks(fontsize=16, rotation=45, ha='right'); plt.yticks(fontsize=16)
    plt.box(False)
    plt.savefig('Images/sub1_1.png', **SAVE_KW)
    plt.savefig('docs/Images/sub1_1.png', **SAVE_KW)
    plt.close(fig)


def barplots_by_country(value_sum, hist6):
    df = pd.DataFrame({'name': value_sum.index, 'value': value_sum.values}).merge(
        pd.DataFrame({'name': hist6.index, 'hist': hist6.values}), on='name', how='left').fillna(0)
    df['diff'] = df['value'] - df['hist']
    def alpha_row(r):
        denom = r['hist'] + 1
        return float(np.clip(abs(r['value'] - r['hist']) / denom / 2 + 0.5, 0, 1))

    # sub2: Top 10 by value
    top10 = df.nlargest(10, 'value').copy().sort_values('value')
    top10['color'] = np.where(top10['value'] > top10['hist'], 'red', 'black')
    top10['alpha'] = top10.apply(alpha_row, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    t10 = top10.reset_index(drop=True)
    sns.barplot(data=t10, x='name', y='value', hue='color', palette={'red':'red','black':'black'}, legend=False, ax=ax)
    # Ensure alpha per bar
    for patch, a in zip(ax.patches, t10['alpha'].tolist()):
        patch.set_alpha(a)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD'); ax.tick_params(bottom=False, left=False)
    plt.yscale('log'); ax.set_yticklabels([]); ax.set_yticks([])
    plt.savefig('Images/sub2.png', **SAVE_KW); plt.close(fig)

    # sub2_d: largest decreases
    dec = df.nsmallest(10, 'diff').copy()
    dec['diff_abs'] = -dec['diff']
    dec['color'] = np.where(dec['diff'] < 0, 'black', 'red')
    dec['alpha'] = dec.apply(alpha_row, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    dd = dec.sort_values('diff_abs').reset_index(drop=True)
    sns.barplot(data=dd, x='name', y='diff_abs', hue='color', palette={'red':'red','black':'black'}, legend=False, ax=ax)
    for patch, a in zip(ax.patches, dd['alpha'].tolist()):
        patch.set_alpha(a)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD'); ax.tick_params(bottom=False, left=False)
    plt.yscale('log'); ax.set_yticklabels([]); ax.set_yticks([])
    plt.savefig('Images/sub2_d.png', **SAVE_KW); plt.close(fig)

    # sub2_i: largest increases
    inc = df.nlargest(10, 'diff').copy()
    inc['color'] = np.where(inc['diff'] > 0, 'red', 'black')
    inc['alpha'] = inc.apply(alpha_row, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    ii = inc.sort_values('diff').reset_index(drop=True)
    sns.barplot(data=ii, x='name', y='diff', hue='color', palette={'red':'red','black':'black'}, legend=False, ax=ax)
    for patch, a in zip(ax.patches, ii['alpha'].tolist()):
        patch.set_alpha(a)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD'); ax.tick_params(bottom=False, left=False)
    plt.yscale('log'); ax.set_yticklabels([]); ax.set_yticks([])
    plt.savefig('Images/sub2_i.png', **SAVE_KW); plt.close(fig)


def _compute_fallback_matches(hist_df: pd.DataFrame, name: str, h_train: int = 10, h: int = 6) -> List[Tuple[float, pd.Series]]:
    """Compute closest historical matches if pickles are missing.
    Returns a list of (distance, series) pairs, best-first.
    """
    if Shape is None or _Finder is None:
        return []
    try:
        # Use the country series
        s = hist_df[name]
        if s.tail(h_train).sum() == 0:
            return []
        sh = Shape(); sh.set_shape(s.tail(h_train))
        # Exclude last h months from search space, mirroring forecast generation
        F = _Finder(hist_df.iloc[:-h, :], sh)
        F.find_patterns(min_d=0.1, select=True, metric='dtw', dtw_sel=2, min_mat=3, d_increase=0.05)
        items = []
        for it in getattr(F, 'sequences', [])[:10]:
            # it expected like (series, dist, ...)
            try:
                series = it[0]; dist = float(it[1]) if len(it) > 1 else 0.0
                if hasattr(series, 'values'):
                    items.append((dist, series))
            except Exception:
                continue
        items.sort(key=lambda x: x[0])
        return items[:4]
    except Exception:
        return []


def _compute_fallback_scenarios(hist_df: pd.DataFrame, name: str, df_conf: Optional[pd.DataFrame], h_train: int = 10, h: int = 6) -> Optional[pd.DataFrame]:
    """Compute scenario trajectories if sce pickle missing, following generate_forecasts logic."""
    if Shape is None or _Finder is None:
        return None
    try:
        s = hist_df[name]
        if s.tail(h_train).sum() == 0:
            return None
        sh = Shape(); sh.set_shape(s.tail(h_train))
        F = _Finder(hist_df.iloc[:-h, :], sh)
        F.find_patterns(min_d=0.1, select=True, metric='dtw', dtw_sel=2, min_mat=3, d_increase=0.05)
        if df_conf is not None:
            F.create_sce(df_conf, h)
            sce_ts = F.val_sce
            return sce_ts
        return None
    except Exception:
        return None


def top4_details(hist, f6, f6_min, f6_max, top4, sce_dict=None, matches_dict=None):
    # For each top-4 country, produce:
    #  - exN.png: last 10 months history
    #  - exN_sce.png: 6‑month forecasts (p50) with optional band between min/max
    #  - exN_all.png: reuse exN.png for simplicity (placeholder for matches)
    hist = hist.set_index('date')
    for idx, name in enumerate(top4, start=1):
        # exN: history last 10 months
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
        if name in hist.columns:
            s = hist[name].tail(10)
            ax.plot(s.index, s.values, marker='o', color='black', linestyle='-', linewidth=3, markersize=6)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        ax.set_frame_on(False)
        plt.tight_layout(); plt.savefig(f'Images/ex{idx}.png', **SAVE_KW); plt.close(fig)
        # Prepare name variants (renamed and original) for pickle lookups
        name_variants = [name]
        if name in RENAME_REV:
            name_variants.append(RENAME_REV[name])
        # exN_sce: scenario mosaic with probabilities (prefer pickles)
        scen_df = None
        if sce_dict:
            for nm in name_variants:
                if nm in sce_dict:
                    val = sce_dict[nm]
                    scen_df = val[1] if isinstance(val, (list, tuple)) and len(val)>1 else None
                    if isinstance(scen_df, pd.DataFrame) and not scen_df.empty:
                        break
        if isinstance(scen_df, pd.DataFrame) and not scen_df.empty:
            num = len(scen_df)
            if num > 2:
                layout = [[0,0,0,0,0,3], [1,1,1,1,1,4], [2,2,2,2,2,5]]
            elif num == 2:
                layout = [[2,2,2,2,2,9],[0,0,0,0,0,3],[0,0,0,0,0,3],[0,0,0,0,0,3],[5,5,5,5,5,6],[1,1,1,1,1,4],[1,1,1,1,1,4],[1,1,1,1,1,4],[7,7,7,7,7,8]]
            else:
                layout = [[1,1,1,1,1,4], [0,0,0,0,0,3], [2,2,2,2,2,5]]
            fig, axes = plt.subplot_mosaic(layout, figsize=(10, 8))
            fig.patch.set_facecolor('white')
            try:
                order_idx = pd.Series(scen_df.index).sort_values(ascending=False).index[:3]
                sel = scen_df.iloc[order_idx, :]
            except Exception:
                sel = scen_df.iloc[:3, :]
            base = hist[name] if name in hist.columns else pd.Series([], dtype=float)
            b = (base - base.min()) / (base.max() - base.min()) if len(base) and (base.max() - base.min()) != 0 else base
            for c, (p, row) in enumerate(sel.iterrows()):
                prob = float(p) if not isinstance(p, tuple) else float(p[0])
                color = '#df2226' if prob >= 0.5 else '#df' + f"{34 + int((0.5 - prob)*100*3):x}" + f"{38 + int((0.5 - prob)*100*3):x}"
                scen = pd.Series(b.tolist() + row.tolist())
                scen = scen * (base.max() - base.min()) + base.min() if len(base) and (base.max()-base.min())!=0 else scen
                ax_line = axes[c]
                ax_text = axes.get(c+3, None)
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
        else:
            # Fallback simple overlay
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            fig.patch.set_facecolor('white'); ax.set_facecolor('white')
            hx = hist[name].tail(10) if name in hist.columns else pd.Series([], dtype=float)
            ax.plot(range(len(hx)), hx.values, color='black', linestyle='-', linewidth=3, marker='o', markersize=6)
            try:
                p50 = f6[name].values
                xs = list(range(len(hx), len(hx) + len(p50)))
                ax.plot(xs, p50, color='red', linestyle='-', linewidth=3, marker='o', markersize=6)
                if f6_min is not None and f6_max is not None and name in f6_min.columns and name in f6_max.columns:
                    lo = f6_min[name].values; hi = f6_max[name].values
                    ax.fill_between(xs, lo, hi, color='red', alpha=0.2)
            except Exception:
                pass
            ax.set_xticks(list(range(len(hx), len(hx)+6)))
            ax.set_xticklabels([f't+{i}' for i in range(1,7)])
            ax.set_frame_on(False); plt.tight_layout(); plt.savefig(f'Images/ex{idx}_sce.png', **SAVE_KW); plt.close(fig)

        # exN_all: closest historical matches
        drawn = False
        if matches_dict:
            try:
                seqs = None
                for nm in name_variants:
                    if nm in matches_dict:
                        seqs = matches_dict[nm]
                        break
                # Expect list of [Series, distance]; pick up to 4 best
                items = []
                if seqs is not None:
                    for it in seqs:
                        if isinstance(it, (list, tuple)) and len(it)>=1:
                            series = it[0]
                            if hasattr(series, 'values'):
                                dist = float(it[1]) if (isinstance(it, (list, tuple)) and len(it)>1) else 0.0
                                items.append((dist, series))
                items.sort(key=lambda x: x[0])
                panel = items[:4]
                if panel:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
                    fig.patch.set_facecolor('white')
                    axes = axes.flatten()
                    for j, (dist, series) in enumerate(panel):
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
                    for j in range(len(panel),4):
                        axes[j].axis('off')
                    plt.tight_layout(); plt.savefig(f'Images/ex{idx}_all.png', **SAVE_KW); plt.close(fig)
                    drawn = True
            except Exception:
                pass
        if not drawn:
            # Compute fallback matches on-the-fly if pickles missing or mismatched
            try:
                matches = _compute_fallback_matches(hist, name)
                if matches:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
                    fig.patch.set_facecolor('white')
                    axes = axes.flatten()
                    for j, (dist, series) in enumerate(matches):
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
                    for j in range(len(matches), 4):
                        axes[j].axis('off')
                    plt.tight_layout(); plt.savefig(f'Images/ex{idx}_all.png', **SAVE_KW); plt.close(fig)
                    drawn = True
            except Exception:
                pass
        if not drawn:
            # default: copy history panel if still nothing
            try:
                from PIL import Image
                img = Image.open(f'Images/ex{idx}.png')
                img.save(f'Images/ex{idx}_all.png')
            except Exception:
                pass


def _resolve_top4_from_best() -> Optional[List[str]]:
    for path in ('best.from_site.csv', 'best.csv'):
        if os.path.exists(path):
            try:
                dfb = pd.read_csv(path)
                if 'name' in dfb.columns:
                    # Files are ordered [4th,3rd,2nd,1st]; return [1st,2nd,3rd,4th]
                    names = dfb['name'].tolist()
                    names = list(reversed(names))
                    return names[:4]
            except Exception:
                continue
    return None


def main():
    ensure_dirs(); setup_fonts()
    f6, f6_min, f6_max = load_forecasts()
    hist = load_hist()
    sce, matches = load_pickles()
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
    # Top-4: prefer site-derived list to align with PDF labels
    top4 = _resolve_top4_from_best() or value_sum.sort_values(ascending=False).head(4).index.tolist()
    top4_details(hist, f6, f6_min, f6_max, top4, sce_dict=sce, matches_dict=matches)
    print('Newsletter images built from forecasts_h6.csv')


if __name__ == '__main__':
    main()
