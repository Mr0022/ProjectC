import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np


def set_seed(seed):
    """Seed every RNG a run draws from: weight init, shuffling and dropout.

    Called once per --itr iteration rather than once per process. Seeding only
    at start-up would let the RNG stream carry over between iterations, so the
    initialisation of run i would depend on how many epochs run i-1 trained for
    before early stopping -- the repeats would then differ partly for reasons
    that have nothing to do with initialisation, which is the spread an itr
    sweep is meant to measure.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize_runs(runs, seeds):
    """Average each test metric over the --itr repeats and print the result.

    The repeats differ only in their seed, so the spread across them is the
    initialisation noise of the configuration -- the number a benchmark table
    should carry, since a single run sits closer to a best case than to a mean.
    Only metrics reported by every repeat are averaged, and nothing is printed
    for a single run, where there is no spread to report.
    """
    if len(runs) < 2:
        return
    keys = [k for k in runs[0] if all(k in r for r in runs)]
    if not keys:
        return

    width = max(len(k) for k in keys)
    lines = [f"  MEAN OVER {len(runs)} RUNS   seeds {seeds[0]}-{seeds[-1]}"]
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        lines.append(f"  {k:<{width}} : {vals.mean():.6f} +/- {vals.std(ddof=1):.6f}"
                     f"   [min {vals.min():.6f}, max {vals.max():.6f}]")
    print("\n" + "=" * 72 + "\n" + "\n".join(lines) + "\n" + "=" * 72)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    # Defaults describe the shipped EUR/USD realized-variance dataset, so a bare
    # `python run.py --task_name long_term_forecast --is_training 1
    #  --model_id x --model DLinear` trains on data/EURUSD-RV.csv.
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='EURUSD-RV.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='RV', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--scale', type=int, default=1,
                        help='standardise the feature columns with a StandardScaler fit on the '
                             'TRAIN split; 1 = on (the Time-Series-Library default), 0 = off. '
                             'Under --log the series is logged first, so the scaler sees ln(RV). '
                             'With --aggregate_mean the target and the forecast are mapped back '
                             'to the RV/ln(RV) scale before the loss and the HAR-comparable '
                             'metrics, so those numbers stay in RV units either way.')
    parser.add_argument('--aggregate_mean', action='store_true', default=False,
                        help='multi-horizon mean aggregation: the model forecasts a SINGLE value, '
                             'the pred_len-day forward average of RV -- mean(RV) in raw mode and '
                             'ln(mean(RV)) under --log. This is exactly the HAR-RV_RUN.PY target '
                             'Y^(h), so set it (with pred_len = h) to compare against HAR-RV.')
    parser.add_argument('--log', action='store_true', default=False,
                        help='train and evaluate on ln(RV) instead of raw RV. The series is logged '
                             'at load time and the horizon target becomes ln(mean(RV)) -- the log '
                             'sits OUTSIDE the mean, matching HAR-RV_RUN.PY --log. Implies '
                             '--drop_nonpositive, since ln needs strictly positive values.')
    parser.add_argument('--drop_nonpositive', action='store_true', default=False,
                        help='drop rows whose target is <= 0 (non-trading days in RV data). Implied '
                             'by --log. Keep it off for datasets where negative targets are '
                             'meaningful, e.g. ETT oil temperature.')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    # Must default to 'avg', as upstream TimeMixer does. TimeMixer is the only
    # model that reads this. With any other value its __multi_scale_process_inputs
    # returns x_enc unchanged -- a [B, T, N] tensor instead of the list of scales
    # forecast() expects -- so zip() then iterates the tensor and yields 2-D
    # slices, and `B, T, N = x.size()` dies with "expected 3, got 2". A default of
    # None made TimeMixer impossible to run without knowing to pass this flag.
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # Repeats are seeded independently: run i uses --seed + i. That keeps every
    # run reproducible on its own, so `--itr N --seed S` and N separate
    # `--itr 1 --seed S+i` jobs (sharded across GPUs) produce the same results.
    parser.add_argument('--itr', type=int, default=1,
                        help='number of repeated runs; run i uses seed (--seed + i)')
    parser.add_argument('--seed', type=int, default=2021,
                        help='base random seed; run i of --itr uses (--seed + i)')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # Patch-based conv models (ModernTCN / TSLANet)
    parser.add_argument('--patch_size', type=int, default=16, help='patch size for ModernTCN/TSLANet')
    parser.add_argument('--patch_stride', type=int, default=8, help='patch stride for ModernTCN')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout for ModernTCN')

    # FITS
    parser.add_argument('--cut_freq', type=int, default=0,
                        help='FITS low-pass cutoff (num of retained freq bins); 0 = auto')

    # WFTNet
    parser.add_argument('--wavelet_scale', type=int, default=4, help='WFTNet CWT scale')
    parser.add_argument('--period_coeff', type=float, default=0.5,
                        help='WFTNet weight balancing wavelet vs fourier branch')

    # AdaWaveNet (adaptive lifting-scheme wavelet network)
    parser.add_argument('--lifting_kernel_size', type=int, default=7, help='conv kernel size of lifting scheme')
    parser.add_argument('--lifting_levels', type=int, default=1, help='levels of lifting scheme')
    parser.add_argument('--regu_details', type=float, default=0.01, help='regu_details of lifting scheme')
    parser.add_argument('--regu_approx', type=float, default=0.01, help='regu_approx of lifting scheme')
    parser.add_argument('--n_clusters', type=int, default=4, help='number of channel clusters for AdaWaveNet')
    parser.add_argument('--sr_ratio', type=int, default=10, help='super resolution ratio (AdaWaveNet)')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()

    # --aggregate_mean means "the target is an average of variances", which only
    # makes sense for a strictly positive series, and --log needs positivity for
    # ln. Either one therefore implies dropping non-positive rows -- the same
    # rule HAR-RV_RUN.PY applies unconditionally. Without this the raw-scale run
    # would train on the zero-RV non-trading days that HAR drops, and the two
    # models would no longer be fitted on the same rows.
    if (args.aggregate_mean or args.log) and not args.drop_nonpositive:
        args.drop_nonpositive = True
        print('Non-positive targets will be dropped (implied by '
              '--aggregate_mean/--log; keeps the sample identical to HAR-RV).')

    # TimeMixer mixes ACROSS scales, so it needs at least two of them: with
    # down_sampling_layers = 0 the scale list holds a single entry and the model
    # dies on `season_list[1]` with a bare IndexError from inside its mixing
    # block. Upstream leaves the default at 0 and relies on every script passing
    # the flag; fail loudly here instead so the cause is obvious.
    if args.model == 'TimeMixer' and args.down_sampling_layers < 1:
        parser.error(
            "TimeMixer is a multi-scale model and needs --down_sampling_layers "
            ">= 1 (upstream uses 3). Re-run with e.g. --down_sampling_layers 3 "
            "--down_sampling_window 2 --down_sampling_method avg.")

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        runs, run_seeds = [], []
        for ii in range(args.itr):
            # Reseed before the model is built so that this repeat is defined
            # entirely by its own seed, and name the run after the seed rather
            # than the loop index so it stays identifiable when the repeats are
            # run as separate jobs.
            seed = args.seed + ii
            set_seed(seed)

            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, seed)
            
            # Override setting for specific model to ensure proper checkpoint naming and logging
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{seed}'

            print('>>>>>>>start training (run {}/{}, seed {}) : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(
                ii + 1, args.itr, seed, setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # Tasks other than long-term forecasting still return None here, in
            # which case there is nothing to average and the summary is skipped.
            run_metrics = exp.test(setting)
            if run_metrics:
                runs.append(run_metrics)
                run_seeds.append(seed)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()

        summarize_runs(runs, run_seeds)
    else:
        # Test-only: rebuild the name of the training run to load, which is now
        # keyed by seed, so --seed selects which repeat is being evaluated.
        set_seed(args.seed)
        exp = Exp(args)  # set experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, args.seed)
        
        # Override setting for specific model to ensure proper checkpoint naming and logging
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{args.seed}'

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
