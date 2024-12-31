#!/bin/bash -ex

#for seed in 0 1 2 3 4
#do
#  CUDA_VISIBLE_DEVICES=$1 python gsimclr_g.py --DS $2 --lr 0.01 --local --num-gc-layers 3 --aug $3 --a $4 --seed $seed

#done

CUDA_VISIBLE_DEVICES=$1 python gsimclr_g.py --DS $2 --lr 0.01 --local --num-gc-layers 3 --aug random2 --a 0.5
