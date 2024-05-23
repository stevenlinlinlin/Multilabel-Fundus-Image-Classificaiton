#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/myconvnext_2layer_wd_concatGAP_rfmid.csv" \
    --transformer_layer 2 \
    --val \
    --weight_decay \
    --normal_class 0 \
    --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext" \
#     --save_results_path "results/myconvnext.csv" \
#     --transformer_layer 2 \
#     --weight_decay \
#     --val \
#     --normal_class 0 \
#     --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "convnext" \
#     --save_results_path "results/convnextV2-L_in22k_384_1.csv" \
#     --val \
#     --normal_class 0 \
#     --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "swin" \
#     --save_results_path "results/swinV2-B_in22k_384_1.csv" \
#     --val \
#     --normal_class 0 \
#     --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "efficientnet" \
#     --save_results_path "results/efficientnetB7_384_dropout_1.csv" \
#     --val \
#     --normal_class 0 \
#     --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "densenet" \
#     --save_results_path "results/densenet_384.csv" \
#     --val \
#     --normal_class 0 \
#     --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "ctran" \
#     --save_results_path "results/ctran_384.csv" \
#     --ctran_model \
#     --val \
#     --normal_class 0 \
#     --dataset "rfmid"