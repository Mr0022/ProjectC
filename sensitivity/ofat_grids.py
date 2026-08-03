"""OFAT sweep grids for the eleven long-term forecasters.

This is the sensitivity twin of ``tuning/search_spaces.py``: where that file
says what Optuna was allowed to SEARCH, this one says what a one-factor-at-a-
time sweep VARIES around the winner it found. The two must agree, or the
sensitivity curves would describe a model that was never tuned -- so every
grid here is the value set of the corresponding ``suggest_*`` call, and every
clamp is the clamp that ``search_spaces.py`` applies.

Two differences follow from the change of purpose.

1. **The grids live in run.py flag space, not in Optuna parameter space.**
   The anchor of a sweep is the ``best_params`` block of a ``<Model>_best.json``
   file, which records the flags the winning command actually passed -- ``d_ff:
   512``, not ``d_ff_mult: 4``; ``stride: 24``, not ``patch_overlap: 'none'``.
   Reconstructing the raw Optuna draw would add a decoding step that can only
   introduce disagreement, so the sweeps are defined directly on the flags.
   Where a flag's range was expressed relative to another (``d_ff`` as a
   multiple of ``d_model``, ModernTCN's ``patch_size`` as a multiple of
   ``patch_stride``, PatchTST's ``stride`` as a fraction of ``patch_len``) the
   grid is rebuilt from the ANCHOR's value of the other knob, so it still
   covers exactly the ratios that were searched.

2. **Continuous knobs get an evenly spaced grid across the searched range**
   -- linear for the ones drawn linearly (``dropout``, ``propalpha``,
   ``period_coeff``), logarithmic for the ones drawn on a log scale
   (``learning_rate``, ``regu_details``, ``regu_approx``). A sweep needs points
   it can plot; the ranges themselves are unchanged.

The anchor value of a swept knob is always added to its grid, so every curve
passes through the tuned configuration and the panels share one reference
point -- the anchor is trained once per model and reused as the centre of
every panel rather than retrained per knob.

WHAT A CURVE MEANS. OFAT measures LOCAL sensitivity: it holds the other
thirteen knobs at the optimum and moves one. It therefore answers "how much
does this knob matter, given that everything else is tuned", which is the
question a reader of a benchmark table asks, and it does NOT answer "what is
the best value of this knob in general" -- a knob can look inert here purely
because a partner knob has already absorbed its effect. Interactions are
invisible to it by construction. That limitation is stated on the figures.

COUPLED KNOBS. Some flags cannot move alone without leaving the searched
space or producing a configuration the architecture rejects. Those are
resolved by :func:`resolve`, which applies exactly the rules
``search_spaces.py`` applies, and the flags it had to move alongside the swept
one are recorded in the results CSV (the ``coupled`` column) so no curve
silently hides a second change.
"""

import math
import os
import sys

try:
    from tuning.search_spaces import FIXED_PROTOCOL, MODELS
except ImportError:  # imported without the repository root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tuning.search_spaces import FIXED_PROTOCOL, MODELS

SEQ_LEN = FIXED_PROTOCOL['seq_len']

# ---------------------------------------------------------------------------
# Shared value sets
# ---------------------------------------------------------------------------
# _optimisation() draws learning_rate log-uniformly on [1e-4, 5e-2]; six points
# at roughly half a decade apart cover it without spending a fifth of the whole
# sweep on one knob. batch_size and lradj are the searched categoricals verbatim.
LEARNING_RATE = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
BATCH_SIZE = [16, 32, 64, 128]
LRADJ = ['type1', 'type3', 'cosine']

# dropout is drawn from U(0.0, 0.3) almost everywhere -- TSLANet is the one
# exception (U(0.05, 0.3)), and gets its own grid below.
DROPOUT = [0.0, 0.1, 0.2, 0.3]

N_HEADS = [2, 4, 8]
D_FF_MULT = (1, 2, 4)
ACTIVATION = ['gelu', 'relu']

# How a knob's axis should be drawn. Anything unlisted is 'ordinal': a small
# ordered set of discrete choices, plotted at even spacing with the values as
# tick labels, because 16/32/64/128 crowded onto a linear axis is unreadable
# and interpolating between them means nothing.
#   'log'     -- continuous, searched on a log scale  -> log x axis
#   'linear'  -- continuous, searched linearly        -> linear x axis
#   'nominal' -- unordered choices                    -> bar panel
KIND = {
    'learning_rate': 'log',
    'regu_details': 'log',
    'regu_approx': 'log',
    'dropout': 'linear',
    'head_dropout': 'linear',
    'propalpha': 'linear',
    'period_coeff': 'linear',
    'lradj': 'nominal',
    'activation': 'nominal',
    'down_sampling_method': 'nominal',
    'decomp_method': 'nominal',
    'channel_independence': 'nominal',
    'use_norm': 'nominal',
}


def kind(param):
    """Axis type for `param` (see KIND)."""
    return KIND.get(param, 'ordinal')


def value_key(value):
    """Stable string key for a swept value.

    The results CSV is the interface between the runner and the plotter, and a
    float that round-trips through str() at one end and float() at the other
    must land on the same grid point, so floats are keyed by repr(). Shared by
    both sides -- never format a value for the CSV any other way.
    """
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _d_ff(anchor):
    """d_ff grid: the searched MULTIPLES of the anchor's d_model.

    The study never drew d_ff on its own -- it drew d_ff_mult in {1, 2, 4} and
    multiplied. Rebuilding the grid from the anchor's width keeps the sweep
    inside the searched region and keeps the quantity that transfers across
    widths, the ratio, as the thing being varied.
    """
    return [anchor['d_model'] * m for m in D_FF_MULT]


# ---------------------------------------------------------------------------
# Per-model grids. Each returns [(flag, values), ...] in panel order; the
# shared optimisation knobs are appended by grids(). A flag the anchor does not
# carry is dropped there, which is how TimeMixer's conditional moving_avg /
# top_k pair resolves to whichever one its winner actually used.
# ---------------------------------------------------------------------------
def _dlinear(a):
    return [('moving_avg', [5, 13, 25, 49])]


def _patchtst(a):
    # stride was searched as patch_overlap in {half, none}, i.e. as one of two
    # fractions of patch_len; the grid is those two values at the anchor's
    # patch length.
    return [
        ('d_model', [16, 32, 64, 128]),
        ('d_ff', _d_ff(a)),
        ('n_heads', N_HEADS),
        ('e_layers', [1, 2, 3]),
        ('patch_len', [8, 16, 24]),
        ('stride', sorted({a['patch_len'] // 2, a['patch_len']})),
        ('dropout', DROPOUT),
        ('activation', ACTIVATION),
    ]


def _itransformer(a):
    return [
        ('d_model', [32, 64, 128, 256]),
        ('d_ff', _d_ff(a)),
        ('n_heads', N_HEADS),
        ('e_layers', [1, 2, 3]),
        ('dropout', DROPOUT),
        ('activation', ACTIVATION),
    ]


def _timesnet(a):
    return [
        ('d_model', [16, 32, 64]),
        ('d_ff', _d_ff(a)),
        ('e_layers', [1, 2, 3]),
        ('top_k', [2, 3, 5]),
        ('num_kernels', [3, 4, 6]),
        ('dropout', DROPOUT),
    ]


def _msgnet(a):
    return [
        ('d_model', [16, 32, 64]),
        ('d_ff', _d_ff(a)),
        ('e_layers', [1, 2]),
        ('n_heads', N_HEADS),
        ('top_k', [2, 3, 5]),
        ('node_dim', [5, 10, 20]),
        ('gcn_depth', [1, 2, 3]),
        ('propalpha', [0.05, 0.15, 0.3, 0.5]),
        ('conv_channel', [8, 16, 32]),
        ('skip_channel', [8, 16, 32]),
        ('dropout', DROPOUT),
    ]


def _timemixer(a):
    return [
        ('d_model', [16, 32, 64]),
        ('d_ff', _d_ff(a)),
        ('e_layers', [1, 2, 3]),
        ('down_sampling_window', [2, 3, 4]),
        ('down_sampling_layers', [1, 2, 3]),
        ('down_sampling_method', ['avg', 'max', 'conv']),
        ('channel_independence', [0, 1]),
        ('use_norm', [0, 1]),
        ('decomp_method', ['moving_avg', 'dft_decomp']),
        ('moving_avg', [13, 25, 49]),   # live only under decomp_method=moving_avg
        ('top_k', [2, 3, 5]),           # live only under decomp_method=dft_decomp
        ('dropout', DROPOUT),
    ]


def _fits(a):
    # cut_freq was searched as an INTEGER over [3, 49] -- the whole rFFT of a
    # 96-step window. Six points spanning it, ending on the full spectrum.
    return [('cut_freq', [3, 6, 12, 24, 36, 49])]


def _wftnet(a):
    return [
        ('d_model', [16, 32, 64]),
        ('d_ff', _d_ff(a)),
        ('e_layers', [1, 2]),
        ('top_k', [2, 3, 5]),
        ('num_kernels', [3, 4, 6]),
        ('wavelet_scale', [3, 4, 5, 6]),
        ('period_coeff', [0.1, 0.3, 0.5, 0.7, 0.9]),
        ('dropout', DROPOUT),
    ]


def _tslanet(a):
    return [
        ('d_model', [32, 64, 128]),
        ('e_layers', [1, 2, 3]),
        ('patch_size', [8, 16, 24, 32]),
        ('dropout', [0.05, 0.1, 0.2, 0.3]),
    ]


def _moderntcn(a):
    # patch_size was searched as patch_stride * mult with mult in {1, 2}, so its
    # grid is those two multiples of the ANCHOR's stride; sweeping patch_stride
    # carries the anchor's multiple along (see _couple_moderntcn).
    stride = a['patch_stride']
    return [
        ('d_model', [16, 32, 64]),
        ('ffn_ratio', [1, 2, 4]),
        ('num_blocks', [1, 2, 3]),
        ('large_size', [13, 21, 31, 51]),
        ('small_size', [3, 5, 7]),
        ('patch_stride', [2, 4, 8, 16]),
        ('patch_size', [stride * m for m in (1, 2)]),
        ('dropout', DROPOUT),
        ('head_dropout', DROPOUT),
    ]


def _adawavenet(a):
    return [
        ('d_model', [16, 32, 64]),
        ('d_ff', _d_ff(a)),
        ('e_layers', [1, 2]),
        ('n_heads', N_HEADS),
        ('lifting_levels', [1, 2, 3]),
        ('lifting_kernel_size', [3, 5, 7, 9]),
        ('regu_details', [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
        ('regu_approx', [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
        ('dropout', DROPOUT),
        ('activation', ACTIVATION),
    ]


_SPECIFIC = {
    'DLinear': _dlinear,
    'PatchTST': _patchtst,
    'iTransformer': _itransformer,
    'TimesNet': _timesnet,
    'MSGNet': _msgnet,
    'TimeMixer': _timemixer,
    'FITS': _fits,
    'WFTNet': _wftnet,
    'TSLANet': _tslanet,
    'ModernTCN': _moderntcn,
    'AdaWaveNet': _adawavenet,
}

assert set(_SPECIFIC) == set(MODELS)


def _shared(a):
    return [('learning_rate', LEARNING_RATE),
            ('batch_size', BATCH_SIZE),
            ('lradj', LRADJ)]


def grids(model, anchor):
    """{flag: [values]} for `model`, anchored at `anchor` (its best_params).

    Panel order: the model's own knobs first, then the three optimisation
    knobs every model shares, so the shared block sits in the same place in
    every figure. The anchor's own value is folded into each grid -- ascending
    for numeric knobs, appended for nominal ones -- so no curve has a hole
    where its centre should be.
    """
    try:
        specific = _SPECIFIC[model]
    except KeyError:
        raise KeyError(f'no OFAT grid for {model!r}; known models: '
                       f"{', '.join(MODELS)}") from None

    out = {}
    for flag, values in specific(anchor) + _shared(anchor):
        if flag not in anchor:
            # A knob this winner never used: TimeMixer's moving_avg when its
            # decomposition is the DFT one, or vice versa. Sweeping it would
            # move a flag the model does not read.
            continue
        out[flag] = _merge_anchor(list(values), anchor[flag], flag)
    return out


def _merge_anchor(values, anchor, flag):
    """Fold the anchor's own value into a knob's grid.

    A continuous knob's anchor is wherever Optuna's log-uniform or uniform draw
    landed -- a learning rate of 0.00114, a dropout of 0.1085 -- which can be a
    hair away from a grid point. Keeping both would spend a five-seed training
    on a configuration indistinguishable from the anchor and then plot the two
    on top of each other, so a grid value that close is REPLACED by the anchor:
    one fifth of a decade on a log axis, 5% of the range on a linear one.
    Discrete knobs were drawn from the grid itself, so their anchor matches a
    value exactly and nothing is dropped.
    """
    if anchor in values:
        return values
    if kind(flag) == 'nominal':
        return values + [anchor]
    if kind(flag) == 'log' and anchor > 0:
        near = lambda v: v > 0 and abs(math.log10(v / anchor)) < 0.08
    else:
        span = max(values) - min(values) or 1.0
        near = lambda v: abs(v - anchor) <= 0.05 * span
    return sorted([v for v in values if not near(v)] + [anchor])


# ---------------------------------------------------------------------------
# Couplings and clamps -- the rules from search_spaces.py, re-applied after an
# OFAT override so a swept point stays inside the searched space AND inside
# what the architecture accepts.
# ---------------------------------------------------------------------------
def _clamp_heads(cfg):
    """n_heads must divide d_model (search_spaces._attention)."""
    n = cfg.get('n_heads')
    if n is None:
        return
    while cfg['d_model'] % n != 0 and n > 1:
        n //= 2
    cfg['n_heads'] = n


def _couple_patchtst(cfg, anchor, param):
    if param == 'patch_len':
        # patch_overlap, not the raw stride, is what was searched: hold the
        # anchor's overlap REGIME and let the stride follow the patch. Holding
        # the stride instead would produce patch_len 8 with stride 24, which
        # skips two thirds of the window -- a configuration no trial ever saw.
        half = anchor['stride'] * 2 == anchor['patch_len']
        cfg['stride'] = cfg['patch_len'] // 2 if half else cfg['patch_len']
    cfg['stride'] = min(cfg['stride'], cfg['patch_len'])
    _clamp_heads(cfg)


def _couple_timemixer(cfg, anchor, param):
    # window**layers must divide seq_len, or the model's inter-level linear
    # maps and the actual pooling lengths part company.
    while (cfg['down_sampling_layers'] > 1
           and SEQ_LEN % (cfg['down_sampling_window'] ** cfg['down_sampling_layers'])):
        cfg['down_sampling_layers'] -= 1

    # decomp_method switches which sub-knob is live. Sweeping it therefore has
    # to hand the other one a value; run.py's defaults (top_k 5, moving_avg 25)
    # are both inside the searched grids, so they are what the swept point uses.
    if cfg.get('decomp_method') == 'dft_decomp':
        cfg.pop('moving_avg', None)
        cfg.setdefault('top_k', 5)
    elif cfg.get('decomp_method') == 'moving_avg':
        cfg.pop('top_k', None)
        cfg.setdefault('moving_avg', 25)


def _couple_moderntcn(cfg, anchor, param):
    if param == 'patch_stride':
        # patch_size was searched as stride * mult, so the anchor's multiple is
        # the thing held fixed while the stride moves.
        mult = max(1, anchor['patch_size'] // anchor['patch_stride'])
        cfg['patch_size'] = cfg['patch_stride'] * mult
    cfg['patch_size'] = max(cfg['patch_size'], cfg['patch_stride'])

    # A kernel wider than twice the patch count is mostly padding, and
    # small_size must not exceed large_size (ReparamLargeKernelConv asserts it).
    patch_num = SEQ_LEN // cfg['patch_stride']
    cfg['large_size'] = min(cfg['large_size'], max(2 * patch_num - 1, 13))
    cfg['small_size'] = min(cfg['small_size'], cfg['large_size'])


_COUPLE = {
    'PatchTST': _couple_patchtst,
    'iTransformer': lambda cfg, a, p: _clamp_heads(cfg),
    'MSGNet': lambda cfg, a, p: _clamp_heads(cfg),
    'AdaWaveNet': lambda cfg, a, p: _clamp_heads(cfg),
    'TimeMixer': _couple_timemixer,
    'ModernTCN': _couple_moderntcn,
}


def resolve(model, anchor, param=None, value=None):
    """The full searched-flag set for one OFAT point.

    Starts from the anchor, overrides `param` with `value` (both None gives the
    anchor itself) and then re-applies the model's clamps, which may move a
    second flag. Callers record whatever moved beyond `param` -- see the
    ``coupled`` column of the results CSV.
    """
    cfg = dict(anchor)
    if param is not None:
        cfg[param] = value
    couple = _COUPLE.get(model)
    if couple is not None:
        couple(cfg, anchor, param)
    return cfg


def coupled_changes(anchor_cfg, cfg, param):
    """Flags that differ from the resolved anchor other than `param`."""
    keys = set(anchor_cfg) | set(cfg)
    return {k: cfg.get(k) for k in sorted(keys)
            if k != param and anchor_cfg.get(k) != cfg.get(k)}
