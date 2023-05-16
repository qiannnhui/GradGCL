## Dependencies


* [torch-geometric](https://github.com/rusty1s/pytorch_geometric) >= 1.6.0
* [ogb](https://github.com/snap-stanford/ogb) == 1.2.4

### JOAO_g Pre-Training & Finetuning: ###

```
./joao_g.sh NCI1 ${gamma} ${a}
```

```gamma``` is tuned from {0.01, 0.1, 1}.


### JOAOv2_g Pre-Training & Finetuning: ###

```
./joaov2_g.sh NCI1 ${gamma} ${a}
```

```gamma``` is tuned from {0.01, 0.1, 1}. JOAOv2 is trained for 40 epochs since multiple projection heads are trained.
```a ``` is the gradient weight between 0 and 1.

## Acknowledgements
- https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised.

- https://github.com/Shen-Lab/GraphCL_Automated/tree/master/unsupervised_TU