#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext" \
    --save_results_path "results/myconvnext_384_plm.csv"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "convnext" \
#     --save_results_path "results/convnextV2-L_in22k_384_1.csv"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "swin" \
#     --save_results_path "results/swinV2-B_in22k_384_1.csv"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "efficientnet" \
#     --save_results_path "results/efficientnetB7_384_dropout_1.csv"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "densenet" \
#     --save_results_path "results/densenet161_384.csv"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "ctran" \
#     --save_results_path "results/ctran_384.csv" \
#     --lr 0.00001 \
#     --ctran_model