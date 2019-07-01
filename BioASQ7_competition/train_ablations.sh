#!/usr/bin/env bash

##############################################

CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 1 1
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 0 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 1 0
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 0 1
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 1 1 1
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 0 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 0 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 0 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 1 1 0

##############################################

CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0111111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1011111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1101111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1110111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111011
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111101
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111110

















