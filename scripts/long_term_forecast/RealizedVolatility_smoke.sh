#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 1-epoch smoke test for all 10 supported long-term-forecasting models on the
# realized-volatility dataset (data/realized_volatility.csv, univariate ln_RV).
#
# Models: DLinear, PatchTST, iTransformer, TimesNet, MSGNet, TimeMixer,
#         FITS, WFTNet, TSLANet, ModernTCN
#
# Runs on CPU by default (drop --no_use_gpu to use CUDA). Each run trains for a
# single epoch and prints the test MSE/MAE — this is a wiring check, not a
# tuned benchmark.
# ---------------------------------------------------------------------------
set -u
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

COMMON=(--task_name long_term_forecast --is_training 1 --data custom
  --root_path ./data/ --data_path realized_volatility.csv --features S --target ln_RV
  --freq d --seq_len 96 --label_len 48 --pred_len 96
  --enc_in 1 --dec_in 1 --c_out 1
  --d_model 32 --d_ff 64 --n_heads 4 --e_layers 2 --d_layers 1
  --top_k 5 --num_kernels 6
  --train_epochs 1 --batch_size 16 --num_workers 0 --learning_rate 0.001
  --des smoke --itr 1 --no_use_gpu)

run_one () {
  local model="$1"; shift
  echo "=============================== $model ==============================="
  python -u run.py --model "$model" --model_id "RV_smoke_${model}" "${COMMON[@]}" "$@"
}

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
