# %%
import os 
import sys 
# sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import time
import multiprocessing
import argparse
from ..data.knn_radius_graph import KNN_Radius_Graph


def arg_parse():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--radius', type=int, default=12)
    parser.add_argument('--threshold', type=int, default=200)
    parser.add_argument('--tissue_id', type=int, default=0)
    parser.add_argument('--celltype_threshold', type=float, default=0.05)
    args = parser.parse_args()
    return args

kidney_annot_3D_tx = pd.read_csv("./data/CosMx_kidney/raw/kidney_annot_3D_tx.csv", index_col=0)
kidney_annot_3D_tx.drop(["z", "x_local_px", "y_local_px"], axis=1, inplace=True)
kidney_annot_3D_tx.rename(columns={'x_global_px': 'x', 'y_global_px':'y', 'z_local_px':'z','target':'gene'}, inplace=True)
# %%
gene_list = []
f = open("./data/CosMx_kidney/raw/gene.txt", 'r')
for gene in f:
    gene_list.append(gene.strip())
f.close()

args = arg_parse()
radius = args.radius
threshold = args.threshold
celltype_threshold = args.celltype_threshold
id = args.tissue_id

value_counts = kidney_annot_3D_tx['cell_ID'].value_counts()
kidney_annot_3D_tx_filtered = kidney_annot_3D_tx[kidney_annot_3D_tx['cell_ID'].isin(value_counts.index[value_counts >= threshold])]
kidney_annot_3D_tx_filtered = kidney_annot_3D_tx_filtered[kidney_annot_3D_tx_filtered['gene'].isin(gene_list)]

value_counts = kidney_annot_3D_tx_filtered['cell_type'].value_counts()
element_proportions = value_counts / len(kidney_annot_3D_tx_filtered)
kidney_annot_3D_tx_filtered = kidney_annot_3D_tx_filtered[kidney_annot_3D_tx_filtered['cell_type'].isin(element_proportions[element_proportions >= celltype_threshold].index)]

# %%


tissue_names = list(kidney_annot_3D_tx_filtered['Run_Tissue_name'].unique())
kidney_annot_3D_tx_filtered = kidney_annot_3D_tx_filtered[kidney_annot_3D_tx_filtered['Run_Tissue_name'] == tissue_names[id]]
tissue_name = tissue_names[id]

save_path = "./data/CosMx_kidney_transfer/" + tissue_name + "/processed_data" 

if not os.path.exists(save_path):
    os.makedirs(save_path)
# %%

def calculate_func(cell_id):
    data_graph = KNN_Radius_Graph(radius=radius, \
        dataset=kidney_annot_3D_tx_filtered, is_3D=True, \
        cell_ID = cell_id, transcript_label="CellComp")
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
cell_id_list = list(kidney_annot_3D_tx_filtered['cell_ID'].unique())[:10000]
pool = multiprocessing.Pool(processes=100)
pool.imap_unordered(calculate_func, cell_id_list)
pool.close()
pool.join()
