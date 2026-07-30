#!/usr/bin/env bash
# =============================================================================
# EUR/USD realized variance -- deep model vs the HAR-RV baseline.
#
# Runs the same model twice over data/EURUSD-RV.csv: once on the raw variance
# scale and once on ln(RV) (--log). Both use --aggregate_mean, which makes the
# network forecast a SINGLE value, the pred_len-day forward average of RV --
# exactly the target HAR-RV_RUN.PY builds:
#
#     raw   :  Y^(h) =     (1/h) * Sum_k RV_(t+k)
#     --log :  Y^(h) = ln( (1/h) * Sum_k RV_(t+k) )
#
# so with pred_len = h the numbers line up with HAR-RV horizon for horizon.
# Compare against:
#     python HAR-RV_RUN.PY              # raw scale
#     python HAR-RV_RUN.PY --log        # log scale
#
# The data flags (--root_path/--data_path/--target/--features/--freq) are the
# defaults now, and are spelled out here only to make the setup explicit.
# =============================================================================

model_name=${MODEL:-DLinear}

for pred_len in 1 5 22; do
  for mode in raw log; do

    if [ "$mode" = "log" ]; then log_flag="--log"; else log_flag=""; fi

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ \
      --data_path EURUSD-RV.csv \
      --data custom \
      --model_id EURUSD_RV_${mode}_h${pred_len} \
      --model $model_name \
      --features S \
      --target RV \
      --freq d \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --aggregate_mean \
      --drop_nonpositive \
      $log_flag \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des "RV_${mode}_h${pred_len}" \
      --itr 1

  done
done
