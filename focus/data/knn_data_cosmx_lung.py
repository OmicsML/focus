# %%
import os 
import sys 
import pandas as pd
import numpy as np
import time
import multiprocessing
import argparse
from knn_radius_graph import KNN_Radius_Graph
from ..data.data_process import NPY2TorchG, subgraph_splits


lung_annot_3D_tx = pd.read_csv("./data/CosMx_lung/raw/lung_annot_3D_tx.csv")
lung_annot_3D_tx.drop(["x_local_px", "y_local_px", "z"], axis=1, inplace=True)
lung_annot_3D_tx.rename(columns={'x_global_px': 'x', 'y_global_px':'y', 'z_local_px':'z', 'target':'gene'}, inplace=True)


def arg_parse():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--radius', type=int, default=12)
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--tissue_id', type=int, default=0)
    args = parser.parse_args()
    return args
gene_list = []
f = open("./data/CosMx_lung_transfer/raw/gene.txt", 'r')
for gene in f:
    gene_list.append(gene.strip())
f.close()

args = arg_parse()
radius = args.radius 
threshold = args.threshold
tissue_id = args.tissue_id
tissue_id = str(tissue_id)

value_counts = lung_annot_3D_tx['cell_ID'].value_counts()
lung_annot_3D_tx_filtered = lung_annot_3D_tx[lung_annot_3D_tx['cell_ID'].isin(value_counts.index[value_counts >= threshold])]
lung_annot_3D_tx_filtered = lung_annot_3D_tx_filtered.loc[lung_annot_3D_tx_filtered['gene'].isin(gene_list)]

lung_annot_3D_tx_filtered['tissue'] = lung_annot_3D_tx_filtered['cell_ID'].str.split("_", expand=True)[1]
lung_annot_3D_tx_filtered = lung_annot_3D_tx_filtered[lung_annot_3D_tx_filtered['tissue'] == tissue_id]
save_path = "./data/CosMx_lung_transfer/tissue" + tissue_id + "/processed_data"
if not os.path.exists(save_path):
    os.makedirs(save_path)
def calculate_func(cell_id):
    data_graph = KNN_Radius_Graph(radius=radius, dataset=lung_annot_3D_tx_filtered, is_3D=True, cell_ID = cell_id, transcript_label="CellComp")
    data_list = []
    cell_ID = data_graph.cell_ID
    edge_index = data_graph.edge_index()
    node_type = data_graph.node_type()
    node_spatial = data_graph.node_spatial()
    graph_label = data_graph.graph_label()
    node_label = data_graph.node_label()
    data_list.append((cell_ID, edge_index, node_type, node_spatial, graph_label,node_label))
    graph_data = np.array(data_list, dtype=object)
    np.save(os.path.join(save_path, f"{cell_id}.npy"), graph_data)
    print(cell_id, "finished")
cell_id_list = list(lung_annot_3D_tx_filtered['cell_ID'].unique())[:100000]
pool = multiprocessing.Pool(processes=100)
pool.imap_unordered(calculate_func, cell_id_list)
pool.close()
pool.join()
