export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# 🔧 IRON数据集专用优化版本
# 结合原始版本的时间嵌入优势 + TSSA注意力机制
# 特别优化IRON数据集的15分钟高频周期性特征
# 使用固定96时间步分别预测96/192/336/720步

echo "=== IRON数据集专用优化版本测试 (固定96输入步) ==="

# IRON数据集 (7特征, 15分钟频率, 非平稳)
# 96 → 96步预测 (24小时) - 中期预测
echo "开始训练: 96 → 96步预测"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_to_96_optimized \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_96to96' \
  --d_model 512 \
  --d_ff 1024 \
  --itr 1 \
  --learning_rate 0.00005 \
  --batch_size 16 \
  --patience 6 \
  --train_epochs 20 \
  --dropout 0.05 \
  --use_time2vec 1

echo "=== 96→96步预测完成 ==="

# IRON数据集 - 96 → 192步预测 (48小时) - 长期预测
echo "开始训练: 96 → 192步预测"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_to_192_optimized \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_96to192' \
  --d_model 768 \
  --d_ff 1536 \
  --itr 1 \
  --learning_rate 0.00003 \
  --batch_size 12 \
  --patience 7 \
  --train_epochs 25 \
  --dropout 0.03 \
  --use_time2vec 1

echo "=== 96→192步预测完成 ==="

# IRON数据集 - 96 → 336步预测 (84小时 = 3.5天) - 超长期预测
echo "开始训练: 96 → 336步预测"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_to_336_optimized \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_96to336' \
  --d_model 768 \
  --d_ff 1536 \
  --itr 1 \
  --learning_rate 0.00002 \
  --batch_size 8 \
  --patience 8 \
  --train_epochs 30 \
  --dropout 0.02 \
  --use_time2vec 1

echo "=== 96→336步预测完成 ==="

# IRON数据集 - 96 → 720步预测 (180小时 = 7.5天) - 极长期预测
echo "开始训练: 96 → 720步预测"
python -u /run.py \
  --is_training 1 \
  --root_path ./dataset/IRON/ \
  --data_path IRON_processed_with_date.csv \
  --model_id IRON_96_to_720_optimized \
  --model $model_name \
  --data IRON \
  --features M \
  --target Usage_kWh \
  --freq 15min \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 5 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'IRON_96to720' \
  --d_model 1024 \
  --d_ff 2048 \
  --itr 1 \
  --learning_rate 0.00001 \
  --batch_size 4 \
  --patience 10 \
  --train_epochs 40 \
  --dropout 0.01 \
  --use_time2vec 1

echo "=== 96→720步预测完成 ==="

