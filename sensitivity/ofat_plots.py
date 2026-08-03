#!/usr/bin/env python3
"""Figures and summary tables for the OFAT sensitivity sweeps.

Reads the long-format CSV written by ``sensitivity/ofat_sensitivity.py`` and
produces, for every metric asked for:

  ofat_<Model>_<metric>.png/pdf   one panel per hyper-parameter: the metric as
                                  that knob moves, mean +/- 1 sd over the seeds,
                                  against the tuned anchor and its seed-noise band
  ofat_tornado_<metric>.png/pdf   how much each knob moves the metric, per model,
                                  ranked, with the seed-noise level marked
  ofat_heatmap_<metric>.png/pdf   the same sensitivities as one models-by-knobs map
  ofat_summary_<metric>.csv       the numbers behind those two figures
  ofat_points_<metric>.csv        every swept point: mean, sd, n seeds

    python sensitivity/ofat_plots.py
    python sensitivity/ofat_plots.py --metrics mse qlike mae --format png pdf
    python sensitivity/ofat_plots.py --models ModernTCN --metrics mse

How to read the panels
----------------------
* The **dashed line and the grey band** are the anchor: the tuned configuration
  trained with the same number of seeds, and +/- 1 sd of that. A curve that
  stays inside the band has not been shown to matter -- the knob moved the
  metric by less than re-seeding does.
* **Every panel of a figure shares its y axis**, so a flat knob looks flat.
  Points too far off the scale to plot are drawn on the boundary as a hollow
  triangle with their value written beside them, rather than being allowed to
  squash every other panel.
* The **star** marks the anchor's own value of that knob. Curves pass through
  it by construction: it is one shared training run, not a per-panel repeat.
* Sensitivity here is **local**. Every other knob is held at the optimum, so a
  panel answers "how much does this knob matter, given that the rest is tuned"
  and not "what is the best value of this knob in general". Interactions are
  invisible to OFAT by construction.
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sensitivity.ofat_grids import MODELS, grids, kind, value_key
from sensitivity.ofat_sensitivity import discover, load_anchor

# --- palette ---------------------------------------------------------------
# One hue carries the data (blue); the anchor is the only second hue, and it is
# also the only star-shaped mark, so identity never rests on colour alone.
BLUE, BLUE_MID, BLUE_PALE = '#2a78d6', '#86b6ef', '#cde2fb'
ANCHOR = '#eb6834'
SURFACE, INK, INK_2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, BASELINE = '#e1e0d9', '#c3c2b7'
SEQUENTIAL = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
              '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
              '#0d366b']
CMAP = LinearSegmentedColormap.from_list('rv_blue', SEQUENTIAL)

METRIC_LABEL = {
    'mse': {'ln_RV': 'MSE  [ln RV]', 'raw_RV': 'MSE  [RV]'},
    'mae': {'ln_RV': 'MAE  [ln RV]', 'raw_RV': 'MAE  [RV]'},
    'qlike': {'ln_RV': 'QLIKE  [RV]', 'raw_RV': 'QLIKE  [RV]'},
    'mse_rv': {'ln_RV': 'MSE  [RV, back-transformed]'},
    'mae_rv': {'ln_RV': 'MAE  [RV, back-transformed]'},
    'val_loss': {'ln_RV': 'validation loss  [ln RV]',
                 'raw_RV': 'validation loss  [RV]'},
}


def _style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'font.size': 8.5,
        'axes.edgecolor': BASELINE,
        'axes.labelcolor': INK_2,
        'axes.titlecolor': INK,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.7,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK_2,
        'ytick.labelcolor': INK_2,
        'legend.frameon': False,
        'figure.dpi': 110,
    })


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def load_results(path):
    df = pd.read_csv(path, dtype={'value': str, 'coupled': str})
    for column in ('value', 'coupled'):
        df[column] = df[column].fillna('')
    return df


def _header(fig, title, subtitle=None, handles=None, legend_cols=5):
    """Two-line header and optional footer legend, measured in INCHES.

    A figure with two panels is a third the height of one with twelve, so a
    header pinned to a fixed figure FRACTION either collides with the panels or
    floats far above them depending on how many knobs a model has. Positions
    are therefore converted from inches, which is what the type actually needs.
    """
    h = fig.get_figheight()
    fig.suptitle(title, fontsize=13, color=INK, y=1 - 0.10 / h, va='top')
    top = 1 - 0.42 / h
    if subtitle:
        fig.text(0.5, 1 - 0.38 / h, subtitle, ha='center', va='top',
                 fontsize=8.5, color=INK_2)
        top = 1 - 0.66 / h
    bottom = 0.0
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=legend_cols,
                   fontsize=8, bbox_to_anchor=(0.5, 0.008))
        bottom = 0.32 / h
    fig.tight_layout(rect=(0, bottom, 1, top))


def anchors_for(results_dir, models):
    """{model: (anchor params, best.json)} for the models present in the CSV."""
    out = {}
    for model, path in discover(results_dir, models):
        _, anchor, _, best = load_anchor(path)
        out[model] = (anchor, best)
    return out


def point_table(df, model, metric, anchor):
    """Per-knob aggregates for one model: {param: DataFrame(value, mean, sd, n)}.

    The anchor is a single training run shared by every panel, so it is spliced
    into each knob's table at that knob's anchor value. Values are ordered by
    the grid that produced them, which is also what puts an ordinal axis in the
    right order (16, 32, 64, 128 -- not the lexicographic 128, 16, 32, 64).
    """
    sub = df[(df['model'] == model)].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors='coerce')
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return {}, None

    base = sub[sub['param'] == 'anchor'][metric]
    anchor_stat = None
    if len(base):
        anchor_stat = {'mean': float(base.mean()),
                       'sd': float(base.std(ddof=1)) if len(base) > 1 else 0.0,
                       'n': int(len(base)),
                       'seeds': base.to_numpy(dtype=float)}

    tables = {}
    model_grids = grids(model, anchor)
    for param, values in model_grids.items():
        rows = sub[sub['param'] == param]
        if rows.empty and anchor_stat is None:
            continue
        order = _axis_order([value_key(v) for v in values],
                            list(rows['value'].unique()))

        records = []
        for key in order:
            if key == value_key(anchor[param]):
                if anchor_stat is None:
                    continue
                records.append({'value': key, 'mean': anchor_stat['mean'],
                                'sd': anchor_stat['sd'], 'n': anchor_stat['n'],
                                'seeds': anchor_stat['seeds'], 'is_anchor': True,
                                'coupled': ''})
                continue
            got = rows[rows['value'] == key]
            if got.empty:
                continue
            seeds = got[metric].to_numpy(dtype=float)
            records.append({'value': key, 'mean': float(seeds.mean()),
                            'sd': float(seeds.std(ddof=1)) if len(seeds) > 1 else 0.0,
                            'n': len(seeds), 'seeds': seeds, 'is_anchor': False,
                            'coupled': str(got['coupled'].iloc[0] or '')})
        if len(records) > 1:
            tables[param] = pd.DataFrame(records)
    return tables, anchor_stat


def _axis_order(grid, recorded):
    """Grid order, plus any recorded value the current grid no longer holds.

    Editing a grid does not invalidate the runs already in the CSV, so a value
    the sweep once trained can outlive its grid. Numeric axes put it back where
    it belongs rather than appending it after the largest value, which would
    draw the point in the wrong place.
    """
    order = list(grid) + [v for v in recorded if v not in grid]
    try:
        return sorted(order, key=float)
    except ValueError:
        return order


def _x_positions(param, table):
    """(x, tick labels, axis scale) for one knob's panel.

    Continuous knobs keep a real numeric axis -- log for the ones the study
    drew on a log scale -- because the distance between two dropout rates is
    meaningful. Discrete choices are placed at even spacing with their values
    as labels: 16/32/64/128 on a linear axis would pile three points into the
    left quarter and say nothing true about the spacing.
    """
    k = kind(param)
    labels = list(table['value'])
    if k in ('log', 'linear'):
        try:
            x = [float(v) for v in labels]
        except ValueError:
            k = 'ordinal'
        else:
            return np.array(x), None, ('log' if k == 'log' else 'linear')
    return np.arange(len(labels), dtype=float), _tick_labels(labels), 'index'


def _log_labels(x, is_anchor, min_gap=0.25):
    """Tick labels for a log axis, thinned so neighbours cannot collide.

    A tuned learning rate lands wherever Optuna put it -- 0.00114 sits a
    twentieth of a decade from the grid's 0.001 -- and printing both writes one
    label over the other. Labels are kept greedily from the left with a minimum
    spacing of `min_gap` decades, except that the anchor always wins its slot:
    it is the one value on the axis the reader has to be able to find.
    """
    keep = []
    for i, v in enumerate(x):
        if not keep:
            keep.append(i)
        elif math.log10(v) - math.log10(x[keep[-1]]) >= min_gap:
            keep.append(i)
        elif is_anchor[i]:
            keep[-1] = i
    return [f'{v:.3g}' if i in keep else '' for i, v in enumerate(x)]


def _tick_labels(labels):
    out = []
    for v in labels:
        try:
            f = float(v)
        except ValueError:
            out.append(str(v))
            continue
        out.append(str(int(f)) if f == int(f) else f'{f:g}')
    return out


def _shared_ylim(tables, anchor_stat):
    """A y range every panel of a figure can share.

    A configuration that diverges -- a learning rate at the top of the searched
    range usually does -- can sit an order of magnitude above the rest, and
    letting it set the scale would flatten every other panel into a straight
    line. The range is therefore built from the points within 2.5x the anchor
    and the outliers are drawn on the boundary instead (see _panel).
    """
    means, sds = [], []
    for table in tables.values():
        means += list(table['mean'])
        sds += list(table['sd'])
    if not means:
        return None
    means, sds = np.array(means), np.array(sds)
    ref = anchor_stat['mean'] if anchor_stat else float(np.median(means))
    keep = means <= ref * 2.5
    if keep.sum() < 2:
        keep = np.ones_like(means, dtype=bool)
    low = float(np.min(means[keep] - sds[keep]))
    high = float(np.max(means[keep] + sds[keep]))
    if anchor_stat:
        low = min(low, anchor_stat['mean'] - anchor_stat['sd'])
        high = max(high, anchor_stat['mean'] + anchor_stat['sd'])
    pad = 0.12 * (high - low or abs(high) or 1.0)
    return low - pad, high + pad


# ---------------------------------------------------------------------------
# Figure 1: per-model OFAT panels
# ---------------------------------------------------------------------------
def _panel(ax, param, table, anchor_stat, ylim):
    x, ticks, scale = _x_positions(param, table)
    mean = table['mean'].to_numpy(dtype=float)
    sd = table['sd'].to_numpy(dtype=float)
    is_anchor = table['is_anchor'].to_numpy(dtype=bool)

    if anchor_stat:                      # the seed-noise band, drawn first
        ax.axhspan(anchor_stat['mean'] - anchor_stat['sd'],
                   anchor_stat['mean'] + anchor_stat['sd'],
                   color=BLUE_PALE, alpha=0.55, lw=0, zorder=0)
        ax.axhline(anchor_stat['mean'], color=ANCHOR, lw=1.1, ls=(0, (4, 3)),
                   zorder=1)

    inside = np.ones_like(mean, dtype=bool)
    if ylim:
        inside = (mean >= ylim[0]) & (mean <= ylim[1])

    # every seed as a faint dot: the spread is the point of the exercise
    for xi, seeds in zip(x, table['seeds']):
        ax.plot(np.full(len(seeds), xi), seeds, ls='none', marker='o', ms=2.6,
                color=MUTED, alpha=0.55, zorder=3)

    if kind(param) == 'nominal':
        # Unordered choices, so no connecting line -- and no bars either: the
        # metric axis is shared with the other panels and does not start at
        # zero, which is exactly the case where a bar's length lies. Dots with
        # the same +/- 1 sd interval as everywhere else say the same thing
        # honestly.
        for mask, marker, size, colour in ((inside & ~is_anchor, 'o', 6.5, BLUE),
                                           (is_anchor, '*', 13, ANCHOR)):
            if mask.any():
                ax.errorbar(x[mask], mean[mask], yerr=sd[mask], fmt=marker,
                            ms=size, color=colour, mec=SURFACE, mew=1.2,
                            ecolor=colour, elinewidth=1.6, capsize=4, zorder=4)
    else:
        ax.fill_between(x, mean - sd, mean + sd, color=BLUE, alpha=0.16, lw=0,
                        zorder=2)
        ax.plot(x, mean, color=BLUE, lw=2.0, zorder=3)
        ax.plot(x[inside & ~is_anchor], mean[inside & ~is_anchor], ls='none',
                marker='o', ms=5.0, color=BLUE, mec=SURFACE, mew=1.4, zorder=4)
        ax.plot(x[is_anchor], mean[is_anchor], ls='none', marker='*', ms=12,
                color=ANCHOR, mec=SURFACE, mew=1.0, zorder=5)

    if ylim:
        # A point off the shared scale is drawn ON the boundary with its value,
        # so a diverging configuration is visible and named without being
        # allowed to set the scale for every other panel.
        ax.set_ylim(*ylim)
        for xi, mi in zip(x[~inside], mean[~inside]):
            over = mi > ylim[1]
            ax.plot([xi], [ylim[1] if over else ylim[0]], marker='^' if over else 'v',
                    ms=6, mfc='none', mec=BLUE, mew=1.4, clip_on=False, zorder=6)
            ax.annotate(f'{mi:.3g}', (xi, ylim[1] if over else ylim[0]),
                        textcoords='offset points',
                        xytext=(0, -11 if over else 9), ha='center',
                        fontsize=6.5, color=BLUE)

    if scale == 'log':
        # Label the swept values themselves; the decade ticks a log axis
        # defaults to would leave four of the six points unlabelled.
        ax.set_xscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(_log_labels(x, is_anchor), fontsize=7, rotation=30,
                           ha='right')
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if ticks is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=7.5)
        ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.grid(axis='x', visible=False)
    ax.set_title(param, fontsize=9, color=INK, pad=4)
    if any(table['coupled']):
        note = sorted({c for c in table['coupled'] if c})
        ax.set_xlabel('with ' + '; '.join(note)[:46], fontsize=6.5, color=MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def figure_panels(model, tables, anchor_stat, metric, scale_name, best, out_base,
                  formats, dpi):
    if not tables:
        return None
    n = len(tables)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.15 * ncols, 2.55 * nrows + 1.0),
                             squeeze=False, sharey=True)
    ylim = _shared_ylim(tables, anchor_stat)

    for ax, (param, table) in zip(axes.ravel(), tables.items()):
        _panel(ax, param, table, anchor_stat, ylim)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for row in range(nrows):
        axes[row][0].set_ylabel(METRIC_LABEL.get(metric, {}).get(scale_name, metric),
                                fontsize=8.5)

    seeds = anchor_stat['n'] if anchor_stat else max(
        int(t['n'].max()) for t in tables.values())
    handles = [
        Line2D([], [], color=BLUE, lw=2.0, marker='o', ms=5, mec=SURFACE, mew=1.2,
               label='swept value: mean over seeds'),
        Patch(facecolor=BLUE, alpha=0.16, label='+/- 1 sd over seeds'),
        Line2D([], [], ls='none', marker='o', ms=2.8, color=MUTED,
               label='individual seeds'),
    ]
    if anchor_stat:      # nothing anchor-shaped is drawn without an anchor run
        handles[2:2] = [
            Line2D([], [], color=ANCHOR, lw=1.1, ls=(0, (4, 3)), marker='*', ms=11,
                   mec=SURFACE, label='tuned anchor'),
            Patch(facecolor=BLUE_PALE, alpha=0.8,
                  label='anchor +/- 1 sd (seed noise)'),
        ]
    _header(fig,
            f'{model} — one-factor-at-a-time hyper-parameter sensitivity',
            f"EUR/USD realised variance · seq_len {best['seq_len']} · "
            f"h = {best['pred_len']} · {seeds} seeds per point · "
            f'every other hyper-parameter held at the tuned optimum',
            handles, legend_cols=min(5, len(handles)))
    return _save(fig, out_base, formats, dpi)


# ---------------------------------------------------------------------------
# Figure 2: tornado ranking
# ---------------------------------------------------------------------------
def figure_tornado(summary, metric, scale_name, out_base, formats, dpi):
    models = [m for m in MODELS if m in set(summary['model'])]
    if not models:
        return None
    ncols = min(3, len(models))
    nrows = math.ceil(len(models) / ncols)
    # Panels share a height, so it is set by the model with the most knobs; a
    # floor of four rows stops a model with two of them drawing slab-like bars.
    rows_max = max(int((summary['model'] == m).sum()) for m in models)
    panel_h = 0.29 * max(rows_max, 4) + 0.85
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(4.6 * ncols, panel_h * nrows + 1.1))

    for index, (ax, model) in enumerate(zip(axes.ravel(), models)):
        rows = summary[summary['model'] == model].sort_values('span_pct')
        y = np.arange(len(rows))
        # No anchor row in the CSV means the noise level is unknown, not zero:
        # leave those bars plain rather than claiming "within noise".
        noise_col = rows['noise_pct'].to_numpy(dtype=float)
        known = np.isfinite(noise_col)
        above = ~known | (rows['span_pct'].to_numpy() > noise_col)
        ax.barh(y, rows['span_pct'], height=0.66,
                color=[BLUE if a else BLUE_PALE for a in above],
                edgecolor=SURFACE, linewidth=1.0, zorder=2,
                hatch=None)
        for yi, a in zip(y, above):       # hatch, so "within noise" is not colour-only
            if not a:
                ax.barh([yi], [rows['span_pct'].to_numpy()[yi]], height=0.66,
                        color='none', edgecolor=BLUE_MID, hatch='///',
                        linewidth=0.0, zorder=3)
        if known.any():
            ax.axvline(float(noise_col[known][0]), color=ANCHOR, lw=1.1,
                       ls=(0, (4, 3)), zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(rows['param'], fontsize=8)
        ax.set_ylim(-0.6, max(len(rows), 4) - 0.4)
        ax.set_title(model, fontsize=10, color=INK, pad=4)
        ax.grid(axis='y', visible=False)
        if index >= len(models) - ncols:      # bottom row of the grid only
            ax.set_xlabel('metric span across the swept grid  '
                          '(% of the tuned value)', fontsize=7.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for yi, v in zip(y, rows['span_pct']):
            ax.annotate(f'{v:.1f}', (v, yi), xytext=(3, 0), fontsize=7,
                        textcoords='offset points', va='center', color=INK_2)
        ax.set_xlim(0, max(1e-9, rows['span_pct'].max()) * 1.18)
    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)

    label = METRIC_LABEL.get(metric, {}).get(scale_name, metric)
    handles = [
        Patch(facecolor=BLUE, label='moves the metric by more than seed noise'),
        Patch(facecolor=BLUE_PALE, edgecolor=BLUE_MID, hatch='///',
              label='within seed noise — no effect demonstrated'),
        Line2D([], [], color=ANCHOR, lw=1.1, ls=(0, (4, 3)),
               label='seed noise: 1 sd of the anchor'),
    ]
    _header(fig, f'Which hyper-parameters move {label}?',
            'local one-factor-at-a-time sensitivity around each model’s tuned '
            'configuration', handles, legend_cols=3)
    return _save(fig, out_base, formats, dpi)


# ---------------------------------------------------------------------------
# Figure 3: models x knobs heatmap
# ---------------------------------------------------------------------------
def figure_heatmap(summary, metric, scale_name, out_base, formats, dpi):
    models = [m for m in MODELS if m in set(summary['model'])]
    grid = summary.pivot_table(index='param', columns='model', values='span_pct')
    grid = grid.reindex(columns=models)
    # rows ordered by how much they move the metric on average, so the strong
    # knobs sit together at the top
    grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]
    if grid.empty:
        return None

    fig, ax = plt.subplots(figsize=(1.05 * len(models) + 3.4,
                                    0.32 * len(grid) + 2.2))
    data = grid.to_numpy(dtype=float)
    vmax = np.nanpercentile(data, 95) if np.isfinite(data).any() else 1.0
    im = ax.imshow(np.ma.masked_invalid(data), cmap=CMAP, aspect='auto',
                   vmin=0, vmax=max(vmax, 1e-9))
    im.cmap.set_bad(GRID)

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(grid)))
    ax.set_yticklabels(grid.index, fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(models)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid)), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=1.4)
    ax.grid(which='major', visible=False)
    ax.tick_params(which='minor', length=0)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '–', ha='center', va='center', fontsize=8, color=MUTED)
                continue
            dark = v > 0.62 * max(vmax, 1e-9)
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7.5,
                    color=SURFACE if dark else INK)

    label = METRIC_LABEL.get(metric, {}).get(scale_name, metric)
    bar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    bar.set_label(f'span of {label} across the swept grid  (% of the tuned value)',
                  fontsize=8, color=INK_2)
    bar.outline.set_visible(False)
    ax.set_title('Local hyper-parameter sensitivity, all models',
                 fontsize=12, color=INK, pad=10)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_base, formats, dpi)


def _save(fig, out_base, formats, dpi):
    written = []
    for ext in formats:
        path = f'{out_base}.{ext}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def summarise(model, tables, anchor_stat, metric):
    """One row per knob: how far it moves the metric, and whether that beats noise.

    span_pct       (max - min) over the knob's grid, as a % of the anchor -- the
                   size of the effect the sweep demonstrates.
    gain_pct       how much better than the anchor the knob's best value is, as a
                   % of the anchor. Zero means the tuned value is the best of the
                   grid; a positive value that clears noise_pct means the search
                   left something on the table along this axis.
    noise_pct      1 sd of the anchor over its seeds, same units. The reference
                   every other number has to be read against.
    """
    rows = []
    ref = anchor_stat['mean'] if anchor_stat else None
    for param, table in tables.items():
        mean = table['mean'].to_numpy(dtype=float)
        best_i = int(np.argmin(mean))
        base = ref if ref not in (None, 0) else float(np.median(mean))
        rows.append({
            'model': model,
            'param': param,
            'n_values': len(table),
            'anchor_value': table.loc[table['is_anchor'], 'value'].iloc[0]
                            if table['is_anchor'].any() else '',
            'anchor_mean': ref,
            'anchor_sd': anchor_stat['sd'] if anchor_stat else np.nan,
            'best_value': table['value'].iloc[best_i],
            'best_mean': float(mean[best_i]),
            'worst_mean': float(mean.max()),
            'span_pct': 100.0 * (mean.max() - mean.min()) / base,
            'gain_pct': 100.0 * max(0.0, (base - mean.min())) / base,
            'noise_pct': 100.0 * (anchor_stat['sd'] / base) if anchor_stat else np.nan,
            'metric': metric,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out['above_noise'] = out['span_pct'] > out['noise_pct']
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Figures for the OFAT sensitivity sweeps')
    ap.add_argument('--csv', default='./sensitivity/ofat_results.csv')
    ap.add_argument('--results_dir', default='./tuning/ProjectC_tuning',
                    help='where the <Model>_best.json anchors live')
    ap.add_argument('--fig_dir', default='./sensitivity/figures')
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--metrics', nargs='+', default=['mse', 'qlike'],
                    choices=['mse', 'mae', 'qlike', 'mse_rv', 'mae_rv', 'val_loss'])
    ap.add_argument('--format', nargs='+', default=['png'],
                    choices=['png', 'pdf', 'svg'])
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        ap.error(f'no results at {args.csv} -- run sensitivity/ofat_sensitivity.py first')

    _style()
    os.makedirs(args.fig_dir, exist_ok=True)
    df = load_results(args.csv)
    present = [m for m in MODELS if m in set(df['model'])]
    if args.models:
        present = [m for m in present if m in args.models]
    if not present:
        ap.error('the results CSV holds none of the requested models')
    anchors = anchors_for(args.results_dir, present)
    scale_name = str(df['scale'].dropna().iloc[0]) if df['scale'].notna().any() else 'ln_RV'

    for metric in args.metrics:
        summaries, points = [], []
        for model in present:
            if model not in anchors:
                print(f'[WARN] {model}: no <Model>_best.json in {args.results_dir}; '
                      f'skipped (the anchor value of each knob is unknown).')
                continue
            anchor, best = anchors[model]
            tables, anchor_stat = point_table(df, model, metric, anchor)
            if not tables:
                continue
            written = figure_panels(
                model, tables, anchor_stat, metric, scale_name, best,
                os.path.join(args.fig_dir, f'ofat_{model}_{metric}'),
                args.format, args.dpi)
            print(f'[{metric}] {model}: {len(tables)} panel(s) -> '
                  f'{", ".join(os.path.basename(p) for p in written)}')
            summaries.append(summarise(model, tables, anchor_stat, metric))
            for param, table in tables.items():
                out = table.drop(columns=['seeds']).copy()
                out.insert(0, 'param', param)
                out.insert(0, 'model', model)
                out['metric'] = metric
                points.append(out)

        if not summaries:
            print(f'[{metric}] nothing to plot.')
            continue
        summary = pd.concat(summaries, ignore_index=True)
        summary_path = os.path.join(args.fig_dir, f'ofat_summary_{metric}.csv')
        summary.to_csv(summary_path, index=False)
        pd.concat(points, ignore_index=True).to_csv(
            os.path.join(args.fig_dir, f'ofat_points_{metric}.csv'), index=False)

        for path in (figure_tornado(summary, metric, scale_name,
                                    os.path.join(args.fig_dir, f'ofat_tornado_{metric}'),
                                    args.format, args.dpi) or []):
            print(f'[{metric}] tornado -> {os.path.basename(path)}')
        for path in (figure_heatmap(summary, metric, scale_name,
                                    os.path.join(args.fig_dir, f'ofat_heatmap_{metric}'),
                                    args.format, args.dpi) or []):
            print(f'[{metric}] heatmap -> {os.path.basename(path)}')
        print(f'[{metric}] tables -> {os.path.basename(summary_path)}, '
              f'ofat_points_{metric}.csv')

    print(f'\nFigures in {args.fig_dir}')


if __name__ == '__main__':
    main()
