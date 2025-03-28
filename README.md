<h1 align="center"> SDCluster: A clustering based self-supervised pre-training method for semantic segmentation of remote sensing images </h1>
<p align="center">

This repository is the official implementation of [SDCluster: A clustering based self-supervised pre-training method for semantic segmentation of remote sensing images](https://www.sciencedirect.com/science/article/abs/pii/S0924271625000796).
<figure align="center">
<img src="figs/Framework.png" width="100%">
</figure>

## Pre-trained Model Weights

### Backbone

|Model Name|Training Dataset|\#params|Download(Google)|Download(Baidu)|
|-|-|-|-|-|
|[Swin Transformer-Small (Swin-S)](cluster_eval/model/backbone/swin_transformer.py)|Million-AID|49M|[Swin-S.pt](https://drive.google.com/file/d/13y7xCuirvglxvVMfYGQ7qOmq4Y6H2BeI/view?usp=sharing)|[Swin-S.pt](https://pan.baidu.com/s/1juYLPc49l-OfVctW1-ROMw?pwd=qkao)|


### Clustering Network

|Model Name|Training Dataset|\#params|Download(Google)|Download(Baidu)|
|-|-|-|-|-|
|[ClusterEval](cluster_eval/model/cluster_eval.py)|Million-AID|55M|[ClusterEval.pt](https://drive.google.com/file/d/1QhuBebahHP6AvSEZ1hcNsZyHuAEJ7GZ3/view?usp=sharing)|[ClusterEval.pt](https://pan.baidu.com/s/1W7pRgOonUymQmfjIPnGvRA?pwd=mho4)|

## Pre-training
The [main_pretrain.py](pretrain/main_pretrain.py) is an example of re-implementing SDCluster pre-training on custom dataset.

## Visualization of Clustering Results
The following are some clustering results, which can be output using the [vis_cluster.py](cluster_eval/vis_cluster.py) by loading our prepared models [ClusterEval.pt](https://drive.google.com/file/d/1QhuBebahHP6AvSEZ1hcNsZyHuAEJ7GZ3/view?usp=sharing).
<figure align="center">
<img src="figs/Clustering Results.png" width="100%">
</figure>

## Dataset
The Shaanxi building extraction dataset can be available at [https://zenodo.org/records/12531251](https://zenodo.org/records/12531251).

## License
This project is for research purpose only.

## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [leopart](https://github.com/MkuuWaUjinga/leopart), and [SlotCon](https://github.com/CVMI-Lab/SlotCon) , whose code has been utilized in this repository.


