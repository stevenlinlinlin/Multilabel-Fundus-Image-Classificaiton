#!/bin/bash

# myconvnext
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext" \
    --save_results_path "results/mured/myconvnext_1layer.csv" \
    --transformer_layer 1 \
    --val \
    --weight_decay \
    --dataset "mured"

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext" \
    --save_results_path "results/mured/myconvnext_2layer.csv" \
    --transformer_layer 2 \
    --val \
    --weight_decay \
    --dataset "mured"

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext" \
    --save_results_path "results/mured/myconvnext_3layer.csv" \
    --transformer_layer 3 \
    --val \
    --weight_decay \
    --dataset "mured"

# myconvnext_concatGAP
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/mured/myconvnext_1layer_concatGAP_warmup.csv" \
    --transformer_layer 1 \
    --val \
    --weight_decay \
    --dataset "mured"

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/mured/myconvnext_2layer_concatGAP_warmup.csv" \
    --transformer_layer 2 \
    --val \
    --weight_decay \
    --dataset "mured"

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/mured/myconvnext_3layer_concatGAP_warmup.csv" \
    --transformer_layer 3 \
    --val \
    --weight_decay \
    --dataset "mured"

# Data Augmentation
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/mured/myconvnext_2layer_concatGAP_ros02.csv" \
    --transformer_layer 2 \
    --val \
    --weight_decay \
    --dataset "mured" \
    --data_aug "ros02" \
    --warmup