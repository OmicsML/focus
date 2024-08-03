# Graph Contrastive Learning of Subcellular-resolution Spatial Transcriptomics Improves Cell Type Annotation and Reveals Critical Molecular Pathways

## requirements:
* torch (=1.11.0)
* networkx (>=2.6.3)
* scikit-learn (>=1.0.2)
* torch-scatter (>=2.0.9)
* torch-sparse (>=0.6.16)
* torch-cluster (>=1.6.0)
* torch-geometric (>=2.1.0)

## Dataset
The offical raw FOCUS dataset is avaiable:
- [[CosMx SMI ]](https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/)
- [[MERFISH MOP ]](https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/)
- [[Xenium DCIS ]](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast)
Please download and put into `data` folder.

## Preprocessing
### 1. knn-based gene neighborhood network generation
```bash
python focus/data/knn_data_cosmx_lung.py # take CosMx lung data as an example.
```
### 2. data split and generation
```bash
python focus/data/build_graph_datasets_lung.py 
```

## Training
```bash
torchrun --nnodes=1 --nproc_per_node=4 run_all.py 
```


