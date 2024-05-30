#!/bin/bash

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/rfmid/myconvnext_2layer_wd_concatGAP.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext" \
#     --save_results_path "results/rfmid/myconvnext.csv" \
#     --transformer_layer 2 \
#     --weight_decay \
#     --val \
#     --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "convnext" \
#     --save_results_path "results/rfmid/convnextV2-L.csv" \
#     --val \
#     --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "swin" \
#     --save_results_path "results/rfmid/swinV2-B.csv" \
#     --val \
#     --dataset "rfmid"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "efficientnet" \
#     --save_results_path "results/rfmid/efficientnetB7.csv" \
#     --val \
#     --dataset "rfmid"

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "densenet" \
    --save_results_path "results/rfmid/densenet161_1.csv" \
    --val \
    --dataset "mured" \
    --data_aug "mlsmote"

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "ctran" \
#     --save_results_path "results/rfmid/ctran.csv" \
#     --ctran_model \
#     --val \
#     --dataset "rfmid" \
#     --lr 0.00001