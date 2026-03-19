export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# 🔧 优化版：IRON dataset - 96 steps prediction (24 hours = 96 * 15min)
# 针对IRON数据集的非平稳性和15分钟高频特性进行优化
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_96_optimized \
  --model iTransformer \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_Optimized' \
  --d_model 512 \
  --d_ff 1024 \
  --itr 1 \
  --learning_rate 0.00005 \
  --batch_size 16 \
  --patience 5 \
  --train_epochs 20 \
  --dropout 0.05 \
  --freq 15min

# 🔧 优化版：IRON dataset - 192 steps prediction (48 hours)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_192_optimized \
  --model iTransformer \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_Optimized' \
  --d_model 512 \
  --d_ff 1024 \
  --itr 1 \
  --learning_rate 0.00005 \
  --batch_size 12 \
  --patience 5 \
  --train_epochs 20 \
  --dropout 0.05 \
  --freq 15min

# 🔧 优化版：IRON dataset - 336 steps prediction (84 hours)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_336_optimized \
  --model iTransformer \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_Optimized' \
  --d_model 768 \
  --d_ff 1536 \
  --itr 1 \
  --learning_rate 0.00003 \
  --batch_size 8 \
  --patience 7 \
  --train_epochs 25 \
  --dropout 0.03 \
  --freq 15min

# 🔧 优化版：IRON dataset - 720 steps prediction (180 hours = 7.5 days)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_720_optimized \
  --model iTransformer \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_Optimized' \
  --d_model 768 \
  --d_ff 1536 \
  --itr 1 \
  --learning_rate 0.00002 \
  --batch_size 6 \
  --patience 7 \
  --train_epochs 30 \
  --dropout 0.02 \
  --freq 15min
