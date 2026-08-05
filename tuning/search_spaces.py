"""Optuna search spaces for the eleven long-term forecasters under models/.

The protocol is fixed by the study and is NOT searched:

    seq_len   = 96      look-back window (business days)
    pred_len  = 1       one-step-ahead forecast
    label_len = 48      decoder start token; none of the eleven models has a
                        real decoder, so it only sizes the (unused) dec_inp

so every space below describes what varies AROUND that protocol: the model's
own capacity/shape knobs plus the optimisation settings.

Three rules keep the spaces honest:

1. Every key a space returns is the name of a run.py flag. A configuration
   found here is therefore reproducible verbatim with `python -u run.py
   --<flag> <value> ...`; nothing can be tuned that cannot be re-run.
2. Only knobs the model actually READS appear. DLinear, for instance, never
   looks at d_model or dropout (its Model takes `individual` as a constructor
   argument, not from configs), so putting them in its space would spend
   trials on values that change nothing.
3. Illegal combinations are CLAMPED, not sampled and rejected. Optuna needs a
   fixed distribution per parameter name -- a categorical whose choices depend
   on an earlier draw raises at the second trial -- so the dependent knob is
   drawn from its full range and then clipped to what the architecture allows.
   Clamping costs a plateau in the response surface; rejection would cost
   whole trials.
4. A range the OFAT sweep has already settled is NARROWED to the region its
   curves single out. sensitivity/ moves one knob at a time around the tuned
   optimum; where a curve showed an arm of a range to be flat, clearly worse,
   or simply never worth a trial, that arm is dropped here so the next study
   spends its budget on the part that actually moves the validation loss. Six
   models carry such a narrowing (DLinear, iTransformer, MSGNet, FITS,
   ModernTCN, AdaWaveNet); the other five keep the original ranges.

Point 4 is the one place where the spaces are deliberately NOT identical
across models. The first round gave every model the same optimiser ranges so
that the benchmark table compared architectures rather than search effort, and
that is still the default -- `_optimisation` keeps the union range unless a
model passes its own bounds. What changed is that the sweep has now MEASURED
where each of those six lives, so holding them to a range their own curves
rule out would no longer be fairness, only wasted trials. Anything not
narrowed by evidence stays shared.

The dataset these ranges are calibrated for is the shipped EUR/USD realized
variance series: ~3.8k daily rows, univariate (features S -> enc_in = 1), of
which ~2.7k windows are training and ~0.5k validation. That is a SMALL sample
for a deep model, so the capacity ranges stay deliberately narrow -- the risk
here is overfitting, not underfitting, and a 512-wide transformer would only
memorise the training split.

Because enc_in = 1, channel-wise knobs are left out on purpose: DLinear's
`individual`, FITS' `individual` and AdaWaveNet's `n_clusters` (which the
model itself clamps to min(n_clusters, enc_in) = 1) are all no-ops on a
univariate series. For a multivariate run they are the first things to add.
"""

# ---------------------------------------------------------------------------
# The protocol, repeated here so a caller can assert against it.
# ---------------------------------------------------------------------------
FIXED_PROTOCOL = {
    'seq_len': 96,
    'pred_len': 1,
    'label_len': 48,
}

# The eleven models, in the order the smoke scripts run them.
MODELS = (
    'DLinear',
    'PatchTST',
    'iTransformer',
    'TimesNet',
    'MSGNet',
    'TimeMixer',
    'FITS',
    'WFTNet',
    'TSLANet',
    'ModernTCN',
    'AdaWaveNet',
)


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------
def _optimisation(trial, learning_rate=(1e-4, 5e-2), batch_sizes=(16, 32, 64, 128)):
    """The three optimiser knobs, shared by default and overridden by evidence.

    THE DEFAULTS are the union range of the whole family: learning_rate
    log-uniform on 1e-4 to 5e-2, batch_size in {16, 32, 64, 128}. The top of
    the learning-rate range exists for DLinear and FITS, which have a few
    hundred parameters and want a far larger step than a transformer; a model
    that has not been swept keeps it, because the deep models simply learn
    within a few trials that the top diverges and TPE stops proposing it. That
    costs a handful of early trials and avoids handing the linear models a
    range their competitors never see.

    THE OVERRIDES are what the OFAT sweep found (see rule 4 in the module
    docstring). Passing bounds here narrows the search for one model only:

        DLinear, AdaWaveNet   lr 1e-3 to 7e-2   the loss keeps falling well
                                                above the old 5e-2 ceiling and
                                                nothing below 1e-3 was ever
                                                competitive
        MSGNet, ModernTCN     lr 1e-3 to 1e-2   a clear interior optimum; both
                                                ends of the union range were
                                                worse
        MSGNet                batch 16-64       128 dropped: with drop_last on
                                                a 519-window validation split
                                                it also discards the most data
        iTransformer          batch 32-128      16 dropped: noisiest and
                                                slowest end of its curve

    batch_size never goes above 128 whatever the override: the train/val
    loaders are built with drop_last=True and the validation split holds only
    a few hundred windows, so a larger batch discards a visible slice of the
    very number the study is minimising.

    lradj is a real knob rather than a formality, and is NOT narrowed for
    anyone -- it stays a three-way choice everywhere. 'type1' HALVES the
    learning rate every epoch, so it effectively caps training at ~8 useful
    epochs whatever --train_epochs says; 'type3' holds the rate for 3 epochs
    then decays by 0.9; 'cosine' anneals over the full budget. Which one wins
    interacts strongly with the learning rate -- a narrowed lr range shifts
    which schedule looks best -- so they are searched together.
    """
    return {
        'learning_rate': trial.suggest_float('learning_rate', *learning_rate, log=True),
        'batch_size': trial.suggest_categorical('batch_size', list(batch_sizes)),
        'lradj': trial.suggest_categorical('lradj', ['type1', 'type3', 'cosine']),
    }


def _d_ff(trial, d_model, choices=(1, 2, 4)):
    """Feed-forward width as a MULTIPLE of d_model.

    Sampling d_ff independently would waste a large part of the grid on
    d_ff < d_model, a bottleneck no one wants, and would make the two
    parameters fight each other in TPE's model. The ratio is the quantity that
    actually transfers across widths.
    """
    return d_model * trial.suggest_categorical('d_ff_mult', list(choices))


def _attention(trial, d_model, head_choices=(2, 4, 8)):
    """n_heads, clamped so that d_model stays divisible by it.

    Every d_model offered below is a multiple of 8, so the clamp is inert for
    the shipped ranges -- it is there so the space survives a narrower
    d_model list.
    """
    n_heads = trial.suggest_categorical('n_heads', list(head_choices))
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads //= 2
    return n_heads


# ---------------------------------------------------------------------------
# 1. DLinear -- decomposition + two linear maps
# ---------------------------------------------------------------------------
def dlinear(trial):
    """One architectural knob: the trend/seasonal split point.

    moving_avg is the kernel of the moving-average decomposition. It MUST be
    odd: the block pads (k-1)//2 on both sides, which restores the original
    length only for odd k -- an even kernel returns L-1 samples and the
    `x - moving_mean` residual then fails to broadcast.

    The values map to trading-time horizons: 13 ~ half a month, 25 ~ one month
    (the library default), 49 ~ one quarter, 73 ~ a third of a year, 97 ~ the
    whole look-back. On realized variance the choice decides how much of the
    slow volatility level is handed to the trend branch, and the OFAT curve
    fell monotonically across the old grid: 5 was the worst point and 49 the
    best, with no sign of having turned. So the short end is dropped and the
    grid is extended PAST 49 to find where it does.

    97 is the sensible ceiling, not an arbitrary stop. The kernel is applied
    with (k-1)//2 replicated samples at each end, so at k = 97 = seq_len + 1
    every output already averages the entire 96-step window; a longer kernel
    only adds more copies of the two endpoints and the trend branch stops
    responding to the interior of the window at all.

    d_model/dropout are absent because DLinear reads neither.

    The learning rate is narrowed to 1e-3 - 7e-2 (see _optimisation): the
    sweep put DLinear's optimum at the top of the old range, which is what a
    few-hundred-parameter linear model should want, and found everything below
    1e-3 flat and slow.
    """
    return {
        'moving_avg': trial.suggest_categorical('moving_avg', [13, 25, 49, 73, 97]),
        **_optimisation(trial, learning_rate=(1e-3, 7e-2)),
    }


# ---------------------------------------------------------------------------
# 2. PatchTST -- channel-independent patch transformer
# ---------------------------------------------------------------------------
def patchtst(trial):
    """Patch geometry is the dominant knob, so it is searched explicitly.

    With seq_len = 96 the patch count is (96 - patch_len)/stride + 2, which is
    also the width of the flatten head: patch_len 8 / stride 4 gives 24
    patches, patch_len 24 / stride 24 gives 5. Short patches see finer
    structure at the price of a much larger head on a small training set.

    `patch_overlap` is sampled instead of a raw stride so that the two stay
    consistent: 'half' is the paper's 50 % overlap, 'none' gives disjoint
    patches. A stride larger than the patch would skip samples entirely.

    factor is not searched: it only parameterises ProbAttention, and PatchTST
    is built with FullAttention, which ignores it.
    """
    d_model = trial.suggest_categorical('d_model', [16, 32, 64, 128])
    patch_len = trial.suggest_categorical('patch_len', [8, 16, 24])
    overlap = trial.suggest_categorical('patch_overlap', ['half', 'none'])
    return {
        'd_model': d_model,
        'd_ff': _d_ff(trial, d_model),
        'n_heads': _attention(trial, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),
        'patch_len': patch_len,
        'stride': patch_len // 2 if overlap == 'half' else patch_len,
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'activation': trial.suggest_categorical('activation', ['gelu', 'relu']),
        **_optimisation(trial),
    }


# ---------------------------------------------------------------------------
# 3. iTransformer -- attention across variates
# ---------------------------------------------------------------------------
def itransformer(trial):
    """Widest d_model of the transformer group, because on a univariate series
    iTransformer degenerates into an MLP over the look-back.

    Its tokens ARE the variates: with enc_in = 1 the encoder holds a single
    token, self-attention over one token is a no-op up to the value/output
    projections, and all the capacity sits in the Linear(96 -> d_model)
    embedding and the FFN. n_heads is kept in the space only because it still
    reshapes those projections; do not expect it to matter here. On a
    multivariate run this is the model whose behaviour changes most.

    batch_size is narrowed to 32-128: the sweep made 16 the worst point of the
    curve by a clear margin. A single-token encoder over an MLP-sized problem
    gets a very noisy gradient from 16 windows, and it is also the slowest
    setting to train, so it costs twice over.
    """
    d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])
    return {
        'd_model': d_model,
        'd_ff': _d_ff(trial, d_model),
        'n_heads': _attention(trial, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'activation': trial.suggest_categorical('activation', ['gelu', 'relu']),
        **_optimisation(trial, batch_sizes=(32, 64, 128)),
    }


# ---------------------------------------------------------------------------
# 4. TimesNet -- FFT periods folded into 2-D Inception blocks
# ---------------------------------------------------------------------------
def timesnet(trial):
    """Cost grows as e_layers * top_k * num_kernels * d_model * d_ff, so the
    ranges are the tightest of the whole file -- this is by far the slowest of
    the eleven.

    top_k is the number of dominant periods pulled out of the rFFT of the
    (seq_len + pred_len) = 97-step series and reshaped into 2-D. Beyond ~5 the
    extra periods are noise on a 97-point spectrum, and each one costs a full
    Inception pass.

    num_kernels is the number of parallel kernel sizes inside each Inception
    block (1x1, 3x3, ... 2k-1); on daily volatility the largest ones already
    span more than a month of the folded period axis.
    """
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    return {
        'd_model': d_model,
        'd_ff': _d_ff(trial, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),
        'top_k': trial.suggest_categorical('top_k', [2, 3, 5]),
        'num_kernels': trial.suggest_categorical('num_kernels', [3, 4, 6]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        **_optimisation(trial),
    }


# ---------------------------------------------------------------------------
# 5. MSGNet -- multi-scale periods + a learned variate graph
# ---------------------------------------------------------------------------
def msgnet(trial):
    """Two families of knobs, and on a univariate series only one of them bites.

    The scale side (top_k periods, d_model, d_ff, n_heads, e_layers) is the
    part that does the work. The graph side (node_dim, gcn_depth, propalpha,
    conv_channel, skip_channel) builds an adjacency over c_out nodes -- one
    node here -- so the softmax over it is identically 1 and the graph carries
    no information. gcn_depth and the channel widths still size the mixprop
    MLP the block runs afterwards, so they are searched, but with small ranges
    and no expectation of a large effect. On a multivariate run these become
    the interesting half.

    e_layers stops at 2: each layer runs the whole multi-scale + graph stack.

    Four knobs move after the sweep, and they move in different directions:

    * d_model drops 16. It was the smallest width offered and the curve was
      still improving at 64, so the bottom of the range was only costing
      trials -- {32, 64} is where the model lives. 128 is deliberately NOT
      added: MSGNet is already the second-heaviest model in the file at
      ~157k grid points of cost, and the training set is ~2.7k windows.
    * skip_channel shifts UP one notch to {16, 32, 64}, the one graph-side
      width whose curve was not flat -- it sizes the skip projection that
      survives even with a single node, so unlike node_dim it does real work
      here.
    * dropout narrows to 0.0-0.2. Above ~0.2 the curve turned up: at this
      width the model is not large enough relative to the sample for heavy
      regularisation to pay for the signal it removes.
    * the learning rate narrows to 1e-3 - 1e-2 and batch_size to {16, 32, 64}
      (see _optimisation).
    """
    d_model = trial.suggest_categorical('d_model', [32, 64])
    return {
        'd_model': d_model,
        'd_ff': _d_ff(trial, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2]),
        'n_heads': _attention(trial, d_model),
        'top_k': trial.suggest_categorical('top_k', [2, 3, 5]),
        'node_dim': trial.suggest_categorical('node_dim', [5, 10, 20]),
        'gcn_depth': trial.suggest_categorical('gcn_depth', [1, 2, 3]),
        'propalpha': trial.suggest_float('propalpha', 0.05, 0.5),
        'conv_channel': trial.suggest_categorical('conv_channel', [8, 16, 32]),
        'skip_channel': trial.suggest_categorical('skip_channel', [16, 32, 64]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.2),
        **_optimisation(trial, learning_rate=(1e-3, 1e-2), batch_sizes=(16, 32, 64)),
    }


# ---------------------------------------------------------------------------
# 6. TimeMixer -- decompose, downsample, mix across scales
# ---------------------------------------------------------------------------
def timemixer(trial):
    """The multi-scale pyramid is the model, so its geometry is the search.

    down_sampling_layers/window define the pyramid: level i is
    seq_len // window**i steps long. The clamp enforces the one hard rule --
    window**layers must DIVIDE seq_len. The model sizes its inter-level linear
    maps as seq_len // window**i (integer division) while the actual pooling
    produces floor-of-the-previous-length, and the two part company as soon as
    a division leaves a remainder: window 3 at 2 layers wants 96//9 = 10 steps
    but pooling 96 -> 32 -> 10 (or 11 for the conv path, which pads) and the
    mixing then fails on a 10-vs-11 mismatch. Verified by construction over
    every (window, layers, method) combination; the surviving pyramids are

        window 2 -> 96, 48, 24, 12   (1-3 layers)
        window 3 -> 96, 32           (1 layer;  9 does not divide 96)
        window 4 -> 96, 24, 6        (1-2 layers; 64 does not divide 96)

    run.py additionally rejects 0 layers outright, since a one-level pyramid
    has nothing to mix and dies on season_list[1].

    decomp_method switches which sub-knob is live, so the two are sampled
    conditionally -- moving_avg (odd, same padding rule as DLinear) for the
    moving-average split, top_k for the DFT split. TPE handles conditional
    parameters natively; a parameter simply goes unrecorded on the trials
    where it played no part.

    channel_independence stays in the space even though enc_in = 1, because it
    selects a genuinely different code path (per-series embedding vs a
    variate-mixing one) rather than just a width.
    """
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    window = trial.suggest_categorical('down_sampling_window', [2, 3, 4])
    layers = trial.suggest_categorical('down_sampling_layers', [1, 2, 3])
    while layers > 1 and FIXED_PROTOCOL['seq_len'] % (window ** layers) != 0:
        layers -= 1

    params = {
        'd_model': d_model,
        'd_ff': _d_ff(trial, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),
        'down_sampling_window': window,
        'down_sampling_layers': layers,
        'down_sampling_method': trial.suggest_categorical(
            'down_sampling_method', ['avg', 'max', 'conv']),
        'channel_independence': trial.suggest_categorical('channel_independence', [0, 1]),
        'use_norm': trial.suggest_categorical('use_norm', [0, 1]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        **_optimisation(trial),
    }

    decomp = trial.suggest_categorical('decomp_method', ['moving_avg', 'dft_decomp'])
    params['decomp_method'] = decomp
    if decomp == 'moving_avg':
        params['moving_avg'] = trial.suggest_categorical('moving_avg', [13, 25, 49])
    else:
        params['top_k'] = trial.suggest_categorical('top_k', [2, 3, 5])
    return params


# ---------------------------------------------------------------------------
# 7. FITS -- one complex-valued linear layer on a low-pass spectrum
# ---------------------------------------------------------------------------
def fits(trial):
    """A single meaningful knob, so it is searched densely rather than as a grid.

    cut_freq is the number of retained rFFT bins of the 96-step window. The
    spectrum has 96//2 + 1 = 49 bins, which is the hard upper bound (the model
    clamps anything larger). Bin b corresponds to a period of 96/b days, so
    cut_freq = 12 keeps everything slower than 8 days, cut_freq = 24 keeps
    everything slower than 4 days, and 48 keeps all but the Nyquist bin. Since
    the whole model is Linear(cut_freq -> cut_freq * (97/96)) in complex space,
    cut_freq is simultaneously the bandwidth AND the parameter count.

    The range is narrowed to 12-48 from the sweep. Below ~12 the curve rose
    sharply: keeping fewer than a dozen bins throws away everything faster
    than a fortnight, and on realized variance that is where the predictable
    short-horizon persistence lives. The top is 48 rather than 49 because the
    49th bin is Nyquist -- the alternating-sign component of a daily series,
    which on this data is noise, and it is the one bin whose removal the sweep
    showed to be free.

    The wide learning-rate ceiling matters here: even at cut_freq = 48 the
    model has a few thousand parameters and trains far faster than any deep
    baseline, so it keeps the full union range.
    """
    return {
        'cut_freq': trial.suggest_int('cut_freq', 12, 48),
        **_optimisation(trial),
    }


# ---------------------------------------------------------------------------
# 8. WFTNet -- wavelet (local) + Fourier (global) branches
# ---------------------------------------------------------------------------
def wftnet(trial):
    """Same Inception backbone as TimesNet plus a CWT branch, hence the same
    tight capacity ranges and e_layers capped at 2 (the shipped smoke script
    runs it at 1).

    period_coeff is the weight between the two branches and is the knob worth
    spending trials on: it says how much of the signal is treated as globally
    periodic (Fourier) versus locally bursty (wavelet). Volatility is mostly
    the latter -- clustered bursts -- so do not be surprised if the optimum
    lands well below 0.5.

    wavelet_scale is the exponent range of the Morlet scales, which are laid
    out as 2**linspace(-1, scale, 8): scale 3 spans ~0.7-8 days, scale 6 spans
    ~0.7-64 days, i.e. up to a quarter. Eight scales are hard-wired by the
    (8, 1) scale convolution, so only their span is tunable.
    """
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    return {
        'd_model': d_model,
        'd_ff': _d_ff(trial, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2]),
        'top_k': trial.suggest_categorical('top_k', [2, 3, 5]),
        'num_kernels': trial.suggest_categorical('num_kernels', [3, 4, 6]),
        'wavelet_scale': trial.suggest_categorical('wavelet_scale', [3, 4, 5, 6]),
        'period_coeff': trial.suggest_float('period_coeff', 0.1, 0.9),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        **_optimisation(trial),
    }


# ---------------------------------------------------------------------------
# 9. TSLANet -- adaptive spectral block over patches
# ---------------------------------------------------------------------------
def tslanet(trial):
    """Four knobs, because TSLANet reads exactly four config fields.

    d_model is the patch embedding width, e_layers the number of
    spectral+ICB blocks, and patch_size the tokenisation -- its stride is
    fixed internally at patch_size // 2, so the patch count is
    (96 - p)/(p/2) + 1: 23 patches at p = 8, down to 5 at p = 32. The output
    head is Linear(d_model * num_patches -> 1), so patch_size and d_model
    jointly set the head size, which dominates the parameter count.

    dropout is doubly loaded: it is both the block dropout AND the ceiling of
    the linearly increasing drop-path schedule across blocks, so its lower
    bound is kept above zero to preserve some stochastic depth.
    """
    return {
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),
        'patch_size': trial.suggest_categorical('patch_size', [8, 16, 24, 32]),
        'dropout': trial.suggest_float('dropout', 0.05, 0.3),
        **_optimisation(trial),
    }


# ---------------------------------------------------------------------------
# 10. ModernTCN -- large-kernel depth-wise convolutions over patches
# ---------------------------------------------------------------------------
def moderntcn(trial):
    """Large kernels are the whole point of the model, so they lead the space.

    Three constraints are enforced by clamping:

    * patch_stride must divide seq_len = 96. The head is sized as
      d_model * (96 // patch_stride) while the stem convolution produces
      floor((96 - stride)/stride) + 1 patches; the two agree only when the
      stride divides 96, otherwise the flatten head sees the wrong width.
    * large_size and small_size must be ODD (the depth-wise convolution pads
      kernel//2, so an even kernel shortens the patch axis and breaks the
      [B, M, D, N] reshape) and small_size <= large_size (asserted by
      ReparamLargeKernelConv).
    * large_size is clipped to twice the patch count. A 51-tap kernel over the
      6 patches produced by stride 16 is 90 % padding -- parameters spent on
      nothing.

    num_blocks stays a single-element list: with use_multi_scale left at its
    default the head is built for the pre-downsampling patch count, so a
    second stage would halve the feature axis and mismatch it. Depth is
    therefore searched WITHIN the one stage.

    Three changes after the sweep:

    * patch_stride drops 2. Forty-eight patches at stride 2 gave the largest
      flatten head of the whole grid and the worst curve with it -- the head
      alone then holds d_model * 48 weights against ~2.7k training windows.
      4-16 is the useful part. Note the knock-on: the largest patch count now
      reachable is 24 (stride 4), so the large_size clamp caps kernels at
      2*24 - 1 = 47 and the 51 choice is a plateau on top of 47 rather than a
      distinct configuration. It is left in the list because the clamp, not
      the list, is what defines the reachable set -- and because a wider
      look-back would make 51 live again without another edit here.
    * dropout moves UP to 0.2-0.6, the one place in this file where a range
      moved away from zero. ModernTCN's curve was still falling at the old
      0.3 ceiling: the model is convolutional and parameter-dense relative to
      the sample, and heavy dropout is what buys it back. head_dropout is
      NOT moved -- it sits on the final projection, its curve was flat, and
      stacking 0.6 on both would starve the head.
    * the learning rate narrows to 1e-3 - 1e-2 (see _optimisation), an
      interior optimum with both ends of the union range clearly worse.
    """
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    patch_stride = trial.suggest_categorical('patch_stride', [4, 8, 16])
    patch_size = patch_stride * trial.suggest_categorical('patch_size_mult', [1, 2])

    patch_num = FIXED_PROTOCOL['seq_len'] // patch_stride
    large = trial.suggest_categorical('large_size', [13, 21, 31, 51])
    large = min(large, max(2 * patch_num - 1, 13))  # 2*n-1 is odd by construction
    small = min(trial.suggest_categorical('small_size', [3, 5, 7]), large)

    return {
        'd_model': d_model,
        'ffn_ratio': trial.suggest_categorical('ffn_ratio', [1, 2, 4]),
        'num_blocks': [trial.suggest_categorical('num_blocks', [1, 2, 3])],
        'large_size': [large],
        'small_size': [small],
        'patch_size': patch_size,
        'patch_stride': patch_stride,
        'dropout': trial.suggest_float('dropout', 0.2, 0.6),
        'head_dropout': trial.suggest_float('head_dropout', 0.0, 0.3),
        **_optimisation(trial, learning_rate=(1e-3, 1e-2)),
    }


# ---------------------------------------------------------------------------
# 11. AdaWaveNet -- learned lifting-scheme wavelet + transformer encoder
# ---------------------------------------------------------------------------
def adawavenet(trial):
    """The lifting depth is the knob that reshapes the model.

    Each lifting level halves the sequence: 96 -> 48 -> 24 -> 12. The inverted
    embedding is built on seq_len // 2**levels, so level 3 already compresses
    the window to 12 steps before the encoder ever sees it. 96 = 2**5 * 3
    tolerates up to 5 levels, but 3 is the deepest that leaves a usable
    resolution on a 96-step daily window.

    regu_details / regu_approx are the lifting regularisers -- they penalise
    detail-coefficient magnitude and the mean drift of the approximation
    branch, which is what keeps the learned wavelet from collapsing to an
    arbitrary invertible map. They are searched on a log scale around the 0.01
    default because their effect is multiplicative, not additive.

    lifting_kernel_size may be even: the scheme pads asymmetrically
    ((k//2, k-1-k//2)), so any k preserves the length -- unlike every other
    convolution in this file.

    n_clusters is absent: the model clamps it to min(n_clusters, enc_in), and
    enc_in = 1 here. sr_ratio and factor are absent too -- the former is read
    only on the super_resolution task, the latter only by ProbAttention.

    The learning rate narrows to 1e-3 - 7e-2, the widest range in the file and
    the only deep model to get one. The lifting scheme is trained jointly with
    the encoder under the two regularisers above, and the sweep showed it
    needs a large step to move the wavelet filters at all -- below 1e-3 the
    learned transform stays near its initialisation and the model degenerates
    into an encoder over a fixed basis. The ceiling goes above the old 5e-2
    because the curve had not turned there.
    """
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    return {
        'd_model': d_model,
        'd_ff': _d_ff(trial, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2]),
        'n_heads': _attention(trial, d_model),
        'lifting_levels': trial.suggest_categorical('lifting_levels', [1, 2, 3]),
        'lifting_kernel_size': trial.suggest_categorical('lifting_kernel_size', [3, 5, 7, 9]),
        'regu_details': trial.suggest_float('regu_details', 1e-3, 1e-1, log=True),
        'regu_approx': trial.suggest_float('regu_approx', 1e-3, 1e-1, log=True),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'activation': trial.suggest_categorical('activation', ['gelu', 'relu']),
        **_optimisation(trial, learning_rate=(1e-3, 7e-2)),
    }


SPACES = {
    'DLinear': dlinear,
    'PatchTST': patchtst,
    'iTransformer': itransformer,
    'TimesNet': timesnet,
    'MSGNet': msgnet,
    'TimeMixer': timemixer,
    'FITS': fits,
    'WFTNet': wftnet,
    'TSLANet': tslanet,
    'ModernTCN': moderntcn,
    'AdaWaveNet': adawavenet,
}

assert set(SPACES) == set(MODELS)


def suggest(model, trial):
    """Draw one configuration for `model`, as a {run.py flag: value} mapping."""
    try:
        space = SPACES[model]
    except KeyError:
        raise KeyError(
            f"no search space for {model!r}; known models: {', '.join(MODELS)}") from None
    return space(trial)
