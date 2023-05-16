#!/bin/bash -ex

CUDA_VISIBLE_DEVICES=$1 python simgrace_g.py --DS $2 --lr 0.01 --local --num-gc-layers 3 --epoch 20 --eta $3 --a $4

