#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --model "densenet" \
    --save_results_path "results/test.csv" \
    # --lr 0.01 \

# python train.py \
#     --model "mydensenet4" \
#     --save_results_path "results/test-1.csv"