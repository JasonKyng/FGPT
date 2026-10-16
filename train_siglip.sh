#!/bin/bash
device=0

num_n=5
num_a=5
top_k=2

epoch=30

# trian on visa
base_dir=train_visa_prompt_pool_${num_n}_${num_a}_${epoch}_top_${top_k[k]}
save_dir=./checkpoints/${base_dir}/
CUDA_VISIBLE_DEVICES=${device} python train_siglip.py \
  --dataset visa \
  --train_data_path /path/to/visa \
  --save_path ${save_dir} \
  --num_n ${num_n} \
  --num_a ${num_a} \
  --top_k ${top_k} \
  --epoch ${epoch} 
 