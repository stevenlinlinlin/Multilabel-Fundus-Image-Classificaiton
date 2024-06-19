#!/bin/bash

# Query2Label
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "q2l" \
#     --save_results_path "results/mured/q2l.csv" \
#     --val \
#     --dataset "mured" \
#     --warmup

# ADD-GCN
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "add_gcn" \
#     --save_results_path "results/mured/add_gcn.csv" \
#     --val \
#     --dataset "mured"

# EfficientNet_v2
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "efficientnet" \
#     --save_results_path "results/mured/efficientnet-v2_warmup.csv" \
#     --val \
#     --dataset "mured" \
#     --warmup

# MaxViT
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "maxvit" \
#     --save_results_path "results/mured/maxvit-b_warmup.csv" \
#     --val \
#     --dataset "mured" \
#     --warmup

# CoAtNet
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "coatnet" \
#     --save_results_path "results/mured/coatnet_warmup.csv" \
#     --val \
#     --dataset "mured" \
#     --warmup

# ViT-L
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "vit" \
#     --save_results_path "results/mured/vit-l_warmup.csv" \
#     --val \
#     --dataset "mured" \
#     --warmup

# Swin-transformer
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "swin" \
#     --save_results_path "results/mured/swinV2-B.csv" \
#     --val \
#     --dataset "mured"

# convnextV2-L
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "convnext" \
#     --save_results_path "results/mured/convnextV2-L.csv" \
#     --val \
#     --dataset "mured"

# myconvnext
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext" \
#     --save_results_path "results/mured/myconvnext_1layer_warmup.csv" \
#     --transformer_layer 1 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext" \
#     --save_results_path "results/mured/myconvnext_2layer_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext" \
#     --save_results_path "results/mured/myconvnext_3layer_warmup.csv" \
#     --transformer_layer 3 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --warmup

# # myconvnext_concatGAP
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_1layer_concatGAP_warmup.csv" \
#     --transformer_layer 1 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_3layer_concatGAP_warmup.csv" \
#     --transformer_layer 3 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --warmup

# # Data Augmentation
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_ros_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --data_aug "ros" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_mlros_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --data_aug "mlros" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_mlsmote_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --data_aug "mlsmote" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_lpros040_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --data_aug "lpros040" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_plm_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --plm \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_remedial_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --data_aug "remedial" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_my_remedial_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --data_aug "my_remedial" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_mlros_remedial_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --data_aug "mlros_remedial" \
#     --warmup

# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "myconvnext_concatGAP" \
#     --save_results_path "results/mured/myconvnext_2layer_concatGAP_mlsmote_remedial_warmup.csv" \
#     --transformer_layer 2 \
#     --val \
#     --weight_decay \
#     --dataset "mured" \
#     --data_aug "mlsmote_remedial" \
#     --warmup

# EfficientNetB7
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "efficientnet" \
#     --save_results_path "results/mured/efficientnetB7.csv" \
#     --val \
#     --weight_decay \
#     --dataset "mured"

# DenseNet161
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "densenet" \
    --save_results_path "results/mured/test.csv" \
    --val \
    --dataset "mured"
    # --loss "asymmetric_loss"

# C-Tran
# CUDA_VISIBLE_DEVICES=$1 python train.py \
#     --model "ctran" \
#     --save_results_path "results/mured/ctran.csv" \
#     --ctran_model \
#     --val \
#     --dataset "mured" \
#     --lr 0.00001