import os 
import numpy as np 
import random
import time
import argparse
import logging
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import json
import torch
from torch import tensor 
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch import optim
from torch_geometric import data
from torch_geometric.transforms import Constant
from torch_geometric.loader import DataLoader
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.append("./focus")
from focus.model.utils import *
from focus.model.model import simclr, get_model_with_default_configs
from focus.model.view_generator import ViewGenerator_subgraph_based_one, ViewGenerator_subgraph_based_pipeline,ViewGenerator_based_one, GIN_MLP_Encoder
from focus.model.losses import *

str2bool = lambda x: x.lower() == "true"
def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch focus semi-supervised learning')
    parser.add_argument('--dataset_name', dest='dataset_name',default="CosMx_lung_10", help='Dataset')
    parser.add_argument('--referency_data_path', type=str, default="./data/MOP/mouse1_sample2/processed/data.pt", help="semi data")
    parser.add_argument('--query_data_path', type=str, default="./data/MOP/mouse1_sample1/processed/data.pt", help="train and test data")
    parser.add_argument('--celltype_mapping', type=str, default="./data/MOP/mouse1_sample1/processed/mapping.json")
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_factor', type=float, default=1.0)
    parser.add_argument('--lr_decay_step_size', type=int, default=20)
    parser.add_argument('--epoch_select', type=str, default='val_max')
    parser.add_argument('--n_layers_gc', dest='num_gc_layers', type=int, default=3, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--n_layers_feat', type=int, default=1)
    parser.add_argument('--n_layers_conv', type=int, default=2)
    parser.add_argument('--n_layers_fc', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--global_pool', type=str, default="sum")
    parser.add_argument('--skip_connection', type=str2bool, default=True)
    parser.add_argument('--res_branch', type=str, default="BNConvReLU")
    parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=False)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--edge_norm', type=str2bool, default=True)
    parser.add_argument('--with_eval_mode', type=str2bool, default=True)
    parser.add_argument('--add_mask', type=str2bool, default=False)
    parser.add_argument('--save_epoch', type=int, default=20, help="save data after how many epochs")
    parser.add_argument('--sparse', type=str2bool, default=True, help="use sparse tensor of subgraph view")
    parser.add_argument('--encoder', type=str, default='ViewGenerator_subgraph_based_pipeline')
    parser.add_argument('--da_policy', type=str, default='None')
    parser.add_argument('--other_setting', type=str, default='None')
    parser.add_argument('--subpool', type=str, default='mean')
    parser.add_argument('--percent', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--aug_ratio', type=float, default=0.2)
    parser.add_argument("--local_rank", type=int)
    return parser.parse_args()


def cl_exp(device,logger, query_dataset, ref_dataset, model_func, epochs, batch_size, lr, lr_decay_factor,
           lr_decay_step_size, weight_decay, epoch_select, args, with_eval_mode=True, add_mask=False,):
    assert epoch_select in ["val_max", "test_max"], epoch_select
    
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    train_accs, test_accs, train_f1s, test_f1s, durations = [], [], [],  [], []
    logger.info("*" * 10)
    
    train_dataset = query_dataset[:9000]
    test_dataset = query_dataset[:9000]
    
    semi_dataset = ref_dataset[:9000]
    
    # load train dataset
    
    semi_sampler = DistributedSampler(semi_dataset)
    train_sampler = DistributedSampler(train_dataset)
    
    semi_loader = DataLoader(semi_dataset, batch_size=batch_size, shuffle=False, sampler=semi_sampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info("Train size: %d" % len(train_dataset))
    logger.info("Semi size: %d" % len(semi_dataset))
    logger.info("Test size: %d" % len(test_dataset))
    dataset = train_dataset+semi_dataset
    num_features = dataset[0].x.shape[-1]
    
    model = model_func(dataset).to(device)

    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, \
        broadcast_buffers=False, find_unused_parameters=True)
    if args.encoder == 'ViewGenerator_subgraph_based_pipeline':
        view_gen1 = ViewGenerator_subgraph_based_pipeline(num_features, args.hidden, GIN_MLP_Encoder, add_mask, args)
        view_gen2 = ViewGenerator_subgraph_based_pipeline(num_features, args.hidden, GIN_MLP_Encoder, add_mask, args)
    elif args.encoder == 'ViewGenerator_subgraph_based_one':
        view_gen1 = ViewGenerator_subgraph_based_one(num_features, args.hidden, GIN_MLP_Encoder, add_mask, args)
        view_gen2 = ViewGenerator_subgraph_based_one(num_features, args.hidden, GIN_MLP_Encoder, add_mask, args)
    elif args.encoder == 'ViewGenerator_based_one':
        view_gen1 = ViewGenerator_based_one(num_features, args.hidden, GIN_MLP_Encoder, args.setting, add_mask, args)
        view_gen2 = ViewGenerator_based_one(num_features, args.hidden, GIN_MLP_Encoder, args.setting, add_mask, args)
    else:
        raise ValueError("Unknown args.encoder %s".format(args.encoder))
        
    view_gen1 = view_gen1.to(device)
    view_gen2 = view_gen2.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    view_optimizer = Adam([ {'params': view_gen1.parameters()},
                            {'params': view_gen2.parameters()} ], lr=lr
                            , weight_decay=weight_decay)

    t_start = time.perf_counter()
    logger.info("*" * 50)
    logger.info("Start training...")
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(1, epochs+1), desc='{}'.format(args.dataset_name)):
        semi_sampler.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        
        # add epoch info on dataset
        for cell in semi_loader.dataset:
            cell.epoch = epoch
        for cell in train_loader.dataset:
            cell.epoch = epoch
        
        train_view_loss, cls_loss, cl_loss = train_node_weight_view_gen_and_cls(
                                                view_gen1, view_gen2,
                                                view_optimizer, 
                                                model, optimizer, 
                                                semi_loader, device, args)
        cl_loss_un = train_node_weight_view_gen_and_cls_unlabel(
                                                view_gen1, view_gen2,
                                                view_optimizer, 
                                                model, optimizer, 
                                                train_loader, device, args)

        train_f1, train_acc = eval_result(model, semi_loader, device, with_eval_mode)
        test_f1, test_acc = eval_result(model, test_loader, device, with_eval_mode)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        
        logger.info('Epoch: {:03d}, Train View Loss: {:.4f}, Cls Loss: {:.4f}, CL Loss: {:.4f}, CL unlabel: {:.4f}, Train Acc: {:.4f}, Train F1-score: {:.4f}, Test Acc: {:.4f}, Test F1-score: {:.4f},'.format(
                                                            epoch, train_view_loss,
                                                            cls_loss, cl_loss, cl_loss_un, train_acc,train_f1, test_acc, test_f1))
        
        if epoch % lr_decay_step_size == 0:
            for param_group in view_optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
                        
        t_end = time.perf_counter()
        durations.append(t_end-t_start)
    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean
        
    

def run_cl_exp(args, device, logger):
    dataset_name = args.dataset_name
    ref_dataset_path = args.referency_data_path
    query_dataset_path = args.query_data_path 
    
    ref_dataset = torch.load(ref_dataset_path)
    query_dataset = torch.load(query_dataset_path)
    celltype_mapping = json.load(open(args.celltype_mapping))
    # load celltype mapping
    for cell in ref_dataset:
        cell.y = celltype_mapping[cell.y]
    for cell in query_dataset:
        cell.y = celltype_mapping[cell.y]
    logger.info("Dataset: {}".format(dataset_name))

    net = "ResGCN"
    model_func = get_model_with_default_configs(model_name = net,
                                                num_feat_layers=args.n_layers_feat,
                                                num_conv_layers=args.n_layers_conv,
                                                num_fc_layers=args.n_layers_fc,
                                                residual = args.skip_connection,
                                                res_branch = args.res_branch,
                                                global_pool = args.global_pool,
                                                dropout = args.dropout,
                                                edge_norm = args.edge_norm,
                                                hidden = args.hidden,)
    
    train_acc, acc, std, duration = cl_exp(
        device,
        logger,
        query_dataset,
        ref_dataset,
        model_func,
        epochs = args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_step_size=args.lr_decay_step_size,
        weight_decay=0,
        epoch_select=args.epoch_select,
        args = args,
        with_eval_mode=args.with_eval_mode)
    summary1 = 'model={}, eval={}'.format(
            net, args.epoch_select)
    summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
        train_acc*100, acc*100, std*100, round(duration, 2))
    logger.info('{}: {}, {}'.format('mid-result', summary1, summary2))


if __name__ == "__main__":
    args = arg_parse()
    set_seed(args.seed)
    device_id = 'cuda:%d' % (args.gpu)
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger()
    logger.info(args)
    
    run_cl_exp(args, device, logger)
    logger.info("Done!")
    