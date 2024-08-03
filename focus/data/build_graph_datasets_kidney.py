import os
import numpy as np
from tqdm import tqdm
import argparse
import sys 
sys.path.append("..") 
import shutil
import multiprocessing
from tqdm import tqdm 
import torch 
from focus.model.utils import *
from focus.data.knn_radius_graph import KNN_Radius_Graph
from focus.data.data_process import NPY2TorchG, subgraph_splits
import warnings
warnings.filterwarnings("ignore")

def arg_parse():
    parser = argparse.ArgumentParser(description='SubcellularGCL')
    parser.add_argument('--tissue_name', type=str, default='Run1080_SP20_10838', help='dataset')
    return parser.parse_args()

args = arg_parse()
tissue_name = args.tissue_name
processed_data_dir = "./data/CosMx_kidney_transfer/" + tissue_name + "/processed_data/"
files = [f for f in os.listdir(processed_data_dir) if f.endswith(".npy")]
data_list = []
for f in tqdm(files):
    data = np.load(os.path.join(processed_data_dir, f), allow_pickle = True)
    data_list.append(data)
data_merged = np.array(data_list)

npy_save_path = "./data/CosMx_kidney_transfer/" + tissue_name + "/raw/"
if not os.path.exists(npy_save_path):
    os.makedirs(npy_save_path)
np.save(npy_save_path + tissue_name + ".npy", data_merged)

data_path = './data/CosMx_kidney_transfer/' + tissue_name + '/'
dataset_name = 'raw/' + tissue_name + '.npy'


source_file = './data/CosMx_kidney/raw/gene.txt'
destination_folder = data_path + 'raw/'
shutil.copy(source_file, destination_folder)


dataset = NPY2TorchG(data_path, dataset_name)
dataset = dataset.load()

one_graph_mask_path = os.path.join(data_path, "one_graph_mask")
one_graph_split_path = os.path.join(data_path, "one_graph_split")
if not os.path.exists(one_graph_mask_path):
    os.makedirs(one_graph_mask_path)
if not os.path.exists(one_graph_split_path):
    os.makedirs(one_graph_split_path)
    
dataset_name = tissue_name

def one_graph_splits_cal(data_id):
    single_graph = data_list[data_id]
    edge_mask, node_group, idx = one_graph_splits_nx(graph=single_graph,idx= data_id)
    torch.save(edge_mask, '{}/one_graph_mask/{}_{}.mat'.format(data_path, dataset_name, data_id))
    torch.save(node_group, '{}/one_graph_split/{}_{}.mat'.format(data_path, dataset_name, data_id))
data_list = dataset

graph_id_list = list(range(len(data_list)))
pool = multiprocessing.Pool(processes=100)
pool.imap_unordered(one_graph_splits_cal, graph_id_list)
pool.close()
pool.join()

# merfe all graphs into one file    
all_graph_mask_path = os.path.join(data_path, "all_graph_mask")
all_graph_split_path = os.path.join(data_path, "all_graph_split")
if not os.path.exists(all_graph_mask_path):
    os.makedirs(all_graph_mask_path)
if not os.path.exists(all_graph_split_path):
    os.makedirs(all_graph_split_path)     
all_graph_mask = []
all_graph_split = []
for i in tqdm(range(len(dataset)), desc="One Graph Saving"):
    try:
        graph_mask = torch.load('{}/one_graph_mask/{}_{}.mat'.format(data_path, dataset_name, i))
        graph_split = torch.load('{}/one_graph_split/{}_{}.mat'.format(data_path, dataset_name, i))
        all_graph_mask.append(graph_mask)
        all_graph_split.append(graph_split)
    except:
        print(i)
        continue
torch.save(all_graph_mask, '{}/all_graph_mask/{}.mat'.format(data_path, dataset_name))
torch.save(all_graph_split, '{}/all_graph_split/{}.mat'.format(data_path, dataset_name)) 

dataset_mask_path = os.path.join(data_path, 'all_graph_mask/{}.mat'.format(dataset_name))
dataset_split_path = os.path.join(data_path, 'all_graph_split/{}.mat'.format(dataset_name))
dataset_mask = np.array(torch.load(dataset_mask_path), dtype=object)
dataset_subsplit = np.array(torch.load(dataset_split_path), dtype=object)

new_dataset = []
for data, data_mask, data_subsplit in tqdm(zip(dataset, dataset_mask, dataset_subsplit)):
    data.intra_edge_mask = data_mask
    data.subsplit = data_subsplit
    data.subsplit_cnt = max(data_subsplit+1)
    new_dataset.append(data)
torch.save(new_dataset, "./data/CosMx_kidney_transfer/" + tissue_name + "/processed/data.pt")