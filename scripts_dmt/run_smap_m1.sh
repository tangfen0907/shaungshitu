python main_dmt.py \
  --mode pretrain \
  --dataset SMAP \
  --data_path ./dataset/SMAP \
  --win_size 100 \
  --input_c 25 \
  --patch_len 10 \
  --batch_size 128 \
  --num_epochs 20 \
  --d_model 128 \
  --n_heads 4 \
  --num_layers 2 \
  --n_memory 20 \
  --lr 1e-3 \
  --lambda_ent 0.01 \
  --temperature 0.1 \
  --topk_ratio 0.05 \
  --anormly_ratio 1.0

python main_dmt.py \
  --mode init_memory \
  --dataset SMAP \
  --data_path ./dataset/SMAP \
  --win_size 100 \
  --input_c 25 \
  --patch_len 10 \
  --batch_size 128 \
  --d_model 128 \
  --n_heads 4 \
  --num_layers 2 \
  --n_memory 20 \
  --temperature 0.1

python main_dmt.py \
  --mode memory_train \
  --dataset SMAP \
  --data_path ./dataset/SMAP \
  --win_size 100 \
  --input_c 25 \
  --patch_len 10 \
  --batch_size 128 \
  --num_epochs 20 \
  --d_model 128 \
  --n_heads 4 \
  --num_layers 2 \
  --n_memory 20 \
  --lr 1e-3 \
  --lambda_ent 0.01 \
  --temperature 0.1 \
  --topk_ratio 0.05 \
  --anormly_ratio 1.0

python main_dmt.py \
  --mode test \
  --dataset SMAP \
  --data_path ./dataset/SMAP \
  --win_size 100 \
  --input_c 25 \
  --patch_len 10 \
  --batch_size 128 \
  --d_model 128 \
  --n_heads 4 \
  --num_layers 2 \
  --n_memory 20 \
  --temperature 0.1 \
  --topk_ratio 0.05 \
  --anormly_ratio 1.0
