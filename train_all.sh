#!/bin/bash

# MSCOCO2014 dataset with ConvNeXtTransformer
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/mured/myconvnext_2layer_concatGAP_voc2012.csv" \
    --transformer_layer 2 \
    --val \
    --weight_decay \
    --dataset "coco2014" \
    --warmup

# VOC2012 dataset with ConvNeXtTransformer
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/mured/myconvnext_2layer_concatGAP_voc2012.csv" \
    --transformer_layer 2 \
    --val \
    --weight_decay \
    --dataset "voc2012" \
    --warmup \
    --save_model

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

# Swin-transformer
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "swin" \
    --save_results_path "results/mured/swinV2-B.csv" \
    --val \
    --dataset "mured"

# convnextV2-L
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "convnext" \
    --save_results_path "results/mured/convnextV2-L.csv" \
    --val \
    --dataset "mured"

# Query2Label
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "q2l" \
    --save_results_path "results/mured/q2l_swinL.csv" \
    --val \
    --dataset "mured"

# MCAR with ResNet101
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "mcar" \
    --save_results_path "results/mured/mcar_resnet101.csv" \
    --val \
    --dataset "mured"

# ADD-GCN
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "add_gcn" \
    --save_results_path "results/mured/add_gcn.csv" \
    --val \
    --dataset "mured"

# C-Tran
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "ctran" \
    --save_results_path "results/mured/ctran.csv" \
    --ctran_model \
    --val \
    --dataset "mured"

# EfficientNet_v2
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "efficientnet" \
    --save_results_path "results/mured/efficientnet-v2.csv" \
    --val \
    --dataset "mured"


# MaxViT
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "maxvit" \
    --save_results_path "results/mured/maxvit-b.csv" \
    --val \
    --dataset "mured"

# CoAtNet
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "coatnet" \
    --save_results_path "results/mured/coatnet.csv" \
    --val \
    --dataset "mured"

# ViT-L
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "vit" \
    --save_results_path "results/mured/vit-l.csv" \
    --val \
    --dataset "mured"

# Data Augmentation
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