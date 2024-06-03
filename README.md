<h1 align="center"> SDCluster: A Cluster based Self-supervised Pre-training Method for Semantic Segmentation of Remote Sensing Images </h1>
<p align="center">
<h4 align="center"> Complete code will be released soon.</h4>


## Pre-trained Model Weights

### Backbone

|Model Name|Training Dataset|\#params|Download|
|-|-|-|-|
|[Swin Transformer-Small (Swin-S)](cluster_eval/model/backbone/swin_transformer.py)|Million-AID|49M|[Swin-S.pt](https://drive.google.com/file/d/13y7xCuirvglxvVMfYGQ7qOmq4Y6H2BeI/view?usp=sharing)|

### Clustering Network

|Model Name|Training Dataset|\#params|Download|
|-|-|-|-|
|[ClusterEval](cluster_eval/model/cluster_eval.py)|Million-AID|55M|[ClusterEval.pt](https://drive.google.com/file/d/1QhuBebahHP6AvSEZ1hcNsZyHuAEJ7GZ3/view?usp=sharing)|

## Visualization of Clustering Results

The following is a visualization of the clustering results, which can be output using the [vis_cluster.py](cluster_eval/vis_cluster.py) by loading our prepared pre-trained models.
<figure align="center">
<img src="figs/Clustering Results.png" width="100%">
</figure>

