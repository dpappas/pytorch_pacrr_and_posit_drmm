#!/usr/bin/env bash

##############################################

CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 1 1

##############################################

CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 1 1

##############################################

CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 0 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 1 0

##############################################

CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 0 1

##############################################

CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 0 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 0 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 0 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 1 1 0

##############################################
















