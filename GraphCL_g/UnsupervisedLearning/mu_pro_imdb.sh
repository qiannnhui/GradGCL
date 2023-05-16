#!/bin/bash -ex

python gsimclr.py --DS MUTAG --lr 0.01 --device cuda:3 --log_interval 4
python gsimclr.py --DS IMDB-BINARY --lr 0.01 --device cuda:3 --log_interval 4
python gsimclr.py --DS PROTEINS --lr 0.01 --device cuda:3 --log_interval 5
