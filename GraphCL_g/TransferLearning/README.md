## Dependencies & Dataset

Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.

## Training & Evaluation
### Pre-training: ###
```
cd ./bio
python pretrain_graphcl_g.py --aug1 random --aug2 random --a 0.5
cd ./chem
python pretrain_graphcl_g.py --aug1 random --aug2 random --a 0.5
```

### Finetuning: ###
```
cd ./bio
./finetune.sh
cd ./chem
./run.sh
```
Results will be recorded in ```result.log```.


## Acknowledgements
https://github.com/snap-stanford/pretrain-gnns.

https://github.com/Shen-Lab/GraphCL/tree/master/transferLearning_MoleculeNet_PPI.