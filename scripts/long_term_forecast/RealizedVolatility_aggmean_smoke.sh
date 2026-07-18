#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Multi-horizon MEAN-AGGREGATION (Option 1) smoke test, all 11 models.
#
# With --aggregate_mean the model forecasts a SINGLE value: the log of the
# horizon-average variance, log(mean(RV)) over the next pred_len days
# (RV = exp(ln_RV), aggregated in variance space via logsumexp). The head is
# built with target_window=1; the loss/metric compare that scalar against the
# aggregated ground truth. This is the volatility-forecasting "average RV over
# the horizon" (HAR) target, not a per-step forecast.
#
# NB: aggregated MSE/MAE are on a smoother target, so the numbers are much
# smaller than the per-step smoke test -- do NOT compare the two scales.
#
# Input stays ln_RV (log space is better-conditioned for volatility models).
# ---------------------------------------------------------------------------
set -u
H=${1:-5}   # horizon (business days): 5 = weekly, 22 = monthly

COMMON=(--task_name long_term_forecast --is_training 1 --data custom
  --root_path ./data/ --data_path realized_volatility.csv --features S --target ln_RV
  --freq d --seq_len 96 --label_len 48 --pred_len "$H" --aggregate_mean
  --enc_in 1 --dec_in 1 --c_out 1
  --d_model 32 --d_ff 64 --n_heads 4 --e_layers 2 --d_layers 1
  --top_k 5 --num_kernels 6
  --train_epochs 1 --batch_size 16 --num_workers 0 --learning_rate 0.001
  --des aggmean --itr 1 --no_use_gpu)

run_one () { local m="$1"; shift; echo "===== $m ====="; python -u run.py --model "$m" --model_id "RV_agg_${m}" "${COMMON[@]}" "$@"; }

run_one DLinear
run_one PatchTST
run_one iTransformer
run_one TimesNet
run_one MSGNet
run_one TimeMixer --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg
run_one FITS
run_one WFTNet --e_layers 1
run_one TSLANet
run_one ModernTCN
run_one AdaWaveNet --lifting_levels 3 --lifting_kernel_size 7 --n_clusters 4
