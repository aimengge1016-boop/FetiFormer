export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer


echo "开始训练: 96 → 96步预测"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_original_96_to_96_optimized \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_original_96to96' \
  --d_model 768 \
  --d_ff 1536 \
  --itr 1 \
  --learning_rate 0.00003 \
  --batch_size 8 \
  --patience 3 \
  --train_epochs 25 \
  --dropout 0.03

echo "=== 96→96步预测完成 ==="

# IRON数据集 - 96 → 192步预测 (48小时) - 长期预测
echo "开始训练: 96 → 192步预测"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_original_96_to_192_optimized \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 5 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_original_96to192' \
  --d_model 1024 \
  --d_ff 2048 \
  --itr 1 \
  --learning_rate 0.00002 \
  --batch_size 6 \
  --patience 3 \
  --train_epochs 30 \
  --dropout 0.02

echo "=== 96→192步预测完成 ==="

# IRON数据集 - 96 → 336步预测 (84小时 = 3.5天) - 超长期预测
echo "开始训练: 96 → 336步预测"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_original_96_to_336_optimized \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 5 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_original_96to336' \
  --d_model 1024 \
  --d_ff 2048 \
  --itr 1 \
  --learning_rate 0.00001 \
  --batch_size 4 \
  --patience 3 \
  --train_epochs 40 \
  --dropout 0.01

echo "=== 96→336步预测完成 ==="

# IRON数据集 - 96 → 720步预测 (180小时 = 7.5天) - 极长期预测
echo "开始训练: 96 → 720步预测"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_original_96_to_720_optimized \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 6 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_original_96to720' \
  --d_model 1280 \
  --d_ff 2560 \
  --itr 1 \
  --learning_rate 0.000005 \
  --batch_size 2 \
  --patience 3 \
  --train_epochs 50 \
  --dropout 0.005

echo "=== 96→720步预测完成 ==="

