export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# IRON dataset - 96 steps prediction (24 hours = 96 * 15min)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_96 \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1

# IRON dataset - 192 steps prediction (48 hours)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_192 \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1

# IRON dataset - 336 steps prediction (84 hours)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_336 \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1

# IRON dataset - 720 steps prediction (180 hours = 7.5 days)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_720 \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1
