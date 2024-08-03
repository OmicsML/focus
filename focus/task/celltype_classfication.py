import sys 

sys.path.insert(0, "..")

import os 
import warnings
import datetime 
import numpy as np
import pandas as pd
import torch 
from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from focus.dataloader._dataloader import FocusDataLoader
from focus.train.trainingplans import FocusTrainingPlan
warnings.filterwarnings("ignore")
import argparse
import pdb 

parser = argparse.ArgumentParser(
    description="Training a model for celltype classification"
)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--batch_size", type=int, default=32)

args = parser.parse_args()

seed_everything(args.seed)

if __name__ == "__main__":
    
    ##################
    # dataset kwargs #
    ##################
    reference_dataset_kwargs = {
        # parameters for reference FocusDataset
        "reference_data_path": "/home/luqiaolin/projects/focus/dataset/kidney_annot_3D_tx_100.csv",
        "gene_list_txt_path": "/home/luqiaolin/projects/focus/dataset/gene_list.txt",
        "knn_graph_radius": None,
        "gene_tx_threshold": 0,
        "celltype_threshold": 0,
        "cell_ID_key": "cell_ID",
        "cell_type_key": "celltype",
        "gene_key": "gene",
        "transcript_key": "subcellular_domains",
        "embedding_key": "one_hot",
        "subcellular_mapping": {"nucleus": 0, "nucleus edge": 1, "cytoplasm": 2, "cell edge": 3, "none": 4},
        "celltype_mapping": None,
    }
    query_dataset_kwargs = {
        # parameters for query FocusDataset
        "query_data_path": "/home/luqiaolin/projects/focus/dataset/kidney_annot_3D_tx_100.csv",
        "gene_list_txt_path": "/home/luqiaolin/projects/focus/dataset/gene_list.txt",
        "knn_graph_radius": None,
        "gene_tx_threshold": 0,
        "celltype_threshold": 0,
        "cell_ID_key": "cell_ID",
        "cell_type_key": "celltype",
        "gene_key": "gene",
        "transcript_key": "subcellular_domains",
        "embedding_type": "one_hot",
        "subcellular_mapping": {"nucleus": 0, "nucleus edge": 1, "cytoplasm": 2, "cell edge": 3, "none": 4},
        "celltype_mapping": "/home/luqiaolin/projects/focus/dataset/mapping.json",   
    }
    dataset_kwargs = {
        # parameters for FocusDataloader
        "reference_data_kwargs": reference_dataset_kwargs,
        "query_data_kwargs": query_dataset_kwargs,
        "train_size": 0.8,
        "batch_size": args.batch_size,
        "shuffle": True,
        "use_cuda": False,
        "num_workers": 0,
        "pin_memory": True,
        # TODO: add more kwargs
    }
    ##################
    # model kwargs #
    ##################
    view_graph_kwargs = {
        "n_feat": 100, # TODO: if use one-hot encoding, this should be the number of unique genes
        "view_graph_dim": 64,
        "view_graph_encoder_s": "GIN_MLP_Encoder",
        "add_mask": False,
    }
    GNN_kwargs = {
        "optimizer": "Adam",
        "optimizer_creator": None,
        "lr": 0.001,
        "gnn_net": "ResGCN",
        "n_layers_fc": 2,
        "n_layers_feat": 2,
        "n_layers_conv": 2,
        "skip_connection": True,
        "res_branch": True,
        "global_pooling": True,
        "dropout": 0.5, 
        "edge_norm": True,
        "hidden": 64,
        "label_num": 10, # TODO: this should be the number of unique celltypes
        "lamb": 0.5,   
    }
    
    model_kwargs = {
        "n_feat": 100, # TODO: if use one-hot encoding, this should be the number of unique genes
        "view_graph_dim": 64,
        "view_graph_encoder_s": "GIN_MLP_Encoder", 
        "add_mask": False,
        "n_feat":100, # TODO: if use one-hot encoding, this should be the number of unique genes
        "optimizer": "Adam",
        "optimizer_creator": None,
        "lr": 0.001,
        "gnn_net": "ResGCN",
        "n_layers_fc": 2,
        "n_layers_feat": 2,
        "n_layers_conv": 2,
        "skip_connection": True,
        "res_branch": True,
        "global_pooling": True,
        "dropout": 0.5, 
        "edge_norm": True,
        "hidden": 64,
        "label_num": 10, # TODO: this should be the number of unique celltypes
        "lamb": 0.5,
        # TODO: add more kwargs
    }
    ##################
    # logger kwargs #
    ##################
    logger_kwargs = {
        "save_dir": "./logs",
        "name": "celltype_classification",
        "version": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "log_graph": True,
        "default_hp_metric": False,
        "description": "",
        "deterministic": False,
        "offline": False,
        "log_model": True,
        "log_train_images": False,
        "log_test_images": False,
        "log_save_interval": 100,
        "log_every_n_steps": 50,
        "flush_logs_every_n_steps": 100,
        "write_to_tb": True,
    }
    
    ##################
    # trainer kwargs #
    ##################
    trainer_kwargs = {
        "strategy": DDPStrategy,
        "gpus": 1,
        "num_nodes": 1,
        "distributed_backend": "ddp",
        "precision": 16,
        "max_epochs": 100,
        "min_epochs": 1
        # TODO: add more kwargs
    }
    
    tb_logger = TensorBoardLogger(**logger_kwargs)  
    dataloader = FocusDataLoader(**dataset_kwargs)
    trainer = Trainer(logger=tb_logger, **trainer_kwargs)
    model_plan = FocusTrainingPlan(**model_kwargs)
    
    trainer.fit(model_plan, dataloader())
    
    