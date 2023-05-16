## Dependencies & Dataset

Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.

## Training & Evaluation
### Step 1: Pre-training: ###
```
cd ./bio
python pretrain_simgrace_g.py --eta 0.1 --a 0.5
cd ./chem
python pretrain_simgrace_g.py --eta 0.1 --a 0.5
```
### Step 2: Finetuning: ###
```
cd ./bio
./finetune.sh
cd ./chem
./run.sh
```
Results will be recorded in ```result.log```.


## Acknowledgements

* https://github.com/snap-stanford/pretrain-gnns.
* https://github.com/Shen-Lab/GraphCL/tree/master/transferLearning_MoleculeNet_PPI.
* https://github.com/junxia97/SimGRACE/tree/main/transfer_learning