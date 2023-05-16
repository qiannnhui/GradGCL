## Dependencies
* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)==1.7.0

## Training & Evaluation

```
./go.sh $GPU_ID $DATASET_NAME $ETA $A
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$ETA``` could be tuned among {0.1, 1.0, 10.0, 100.0}.
```$A``` is the gradient weight.
## Acknowledgements
- https://github.com/Shen-Lab/GraphCL/tree/master/unsupervised_TU

- https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised.

- https://github.com/junxia97/SimGRACE/tree/main/unsupervised_TU