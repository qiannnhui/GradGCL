###
Please refer to the original code.
https://github.com/nerdslab/bgrl

### Setup

To set up a Python virtual environment with the required dependencies, run:
```
python3 -m venv bgrl_env
source bgrl_env/bin/activate
pip install --upgrade pip
```

Follow instructions to install 
[PyTorch 1.9.1](https://pytorch.org/get-started/locally/) and 
[PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html):
```
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install absl-py==0.12.0 tensorboard==2.6.0 ogb
```

The code uses PyG (PyTorch Geometric).
All datasets are available through this package.


## Experiments on transductive tasks

### Train model from scratch
```bash
python3 train_transductive.py --flagfile=config/coauthor-cs.cfg
```

Flags can be overwritten:
```bash
python3 train_transductive.py --flagfile=config/coauthor-cs.cfg\
                              --logdir=./runs/coauthor-cs-256\
                              --predictor_hidden_size=256
```
Test accuracies under linear evaluation are reported on TensorBoard. 
To start the tensorboard server run the following command:
```bash
tensorboard --logdir=./runs
```

