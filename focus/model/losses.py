import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import os 
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pickle as pkl

def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise ValueError('Unknown measure: {}'.format(measure))

    if average:
        return Ep.mean()
    else:
        return Ep

def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise ValueError('Unknown measure: {}'.format(measure))

    if average:
        return Eq.mean()
    else:
        return Eq

def local_global_loss_(l_enc, g_enc, edge_index, batch, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos

def adj_loss_(l_enc, g_enc, edge_index, batch):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    adj = torch.zeros((num_nodes, num_nodes)).cuda()
    mask = torch.eye(num_nodes).cuda()
    for node1, node2 in zip(edge_index[0], edge_index[1]):
        adj[node1.item()][node2.item()] = 1.
        adj[node2.item()][node1.item()] = 1.

    res = torch.sigmoid((torch.mm(l_enc, l_enc.t())))
    res = (1-mask) * res
    loss = nn.BCELoss()(res, adj)
    return loss



def loss_cl(x1, x2):
    T = 0.5
    batch_size, _ = x1.shape
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    sim_matrix_a = torch.einsum('ik, jk-> ij', x1, x2) / torch.einsum('i, j-> ij', x1_abs, x2_abs)
    sim_matrix_a = torch.exp(sim_matrix_a / T)
    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
    loss_a = - torch.log(loss_a).mean()
    
    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
    sim_matrix_b = torch.exp(sim_matrix_b / T)
    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
    loss_b = - torch.log(loss_b).mean()

    loss = (loss_a + loss_b) / 2
    return loss

def train_cl(view_gen1, view_gen2, view_optimizer, model, optimizer, data_loader, device):
    loss_all = 0
    model.train()
    total_graphs = 0
    for data in data_loader:
        view_optimizer.zero_grad()
        optimizer.zero_grad()
        data = data.to(device)
        _, view1 = view_gen1(data, True)
        _, view2 = view_gen2(data, True)
        
        out1 = model(view1)
        out2 = model(view2)
        loss = loss_cl(out1, out2)
        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        loss.backward()

        optimizer.step()
        view_optimizer.step()

    loss_all /= total_graphs
    return loss_all

def train_cl_with_sim_loss(view_gen1, view_gen2, view_optimizer, model, optimizer, data_loader, device, args=None):
    loss_all = 0
    model.train()
    total_graphs = 0
    for data in data_loader:
        optimizer.zero_grad()
        view_optimizer.zero_grad()
        if len(data.subsplit_cnt) == 1:
            continue
        if len(set(data.y)) == 1:
            continue
        data = data.to(device)
        view1, _ = view_gen1(data, True)
        view2, _ = view_gen2(data, True)
        input_list = [data, view1, view2]
        input1, input2 = random.choices(input_list, k=2)
        out1 = model(input1, device)
        out2 = model(input2, device)
        
        cl_loss = loss_cl(out1, out2)
        loss = cl_loss

        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        loss.backward()    
        optimizer.step()
        view_optimizer.step()
    loss_all /= total_graphs
    return loss_all

def train_node_weight_view_gen_and_cls(view_gen1, view_gen2, view_optimizer, 
                                        model, optimizer, loader, device, args=None):
    view_gen1.train()
    view_gen2.train()
    model.train()

    loss_all = 0
    cls_loss_all = 0
    cl_loss_all = 0
    total_graphs = 0
    
    alpha = args.alpha
    
    data_list = []
    
    for i, data in enumerate(loader):
        view_optimizer.zero_grad()
        optimizer.zero_grad()
        
        data = data.to(device)
        
        if args.encoder == 'ViewGenerator_subgraph_based_pipeline':
            view1, _ = view_gen1(data, True)
            view2, _ = view_gen2(data, True)
        elif args.encoder == 'ViewGenerator_subgraph_based_one':
            view1, _ = view_gen1(data, True)
            view2, _ = view_gen2(data, True)
        elif args.encoder == 'ViewGenerator_based_one':
            view1, aug_data = view_gen1(data, True)
            view2, _ = view_gen2(data, True)
            data_list.extend(aug_data)
        else:
            raise ValueError("Unknown args.encoder %s".format(args.encoder))

        output = model(data)
        output1 = model(view1)
        output2 = model(view2)        

        loss0 = F.nll_loss(output, data.y)
        loss1 = F.nll_loss(output1, data.y)
        loss2 = F.nll_loss(output2, data.y)
        
        cl_loss = loss_cl(output1, output2)  # contrastive learning loss
        cls_loss = (loss0 + loss1 + loss2) / 3 # classification loss
        
        loss =  cl_loss + alpha * cls_loss
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs
        cl_loss_all += cl_loss.item() * data.num_graphs

        total_graphs += data.num_graphs
        view_optimizer.step()
        optimizer.step()
        
    if data.epoch[0].item() >= args.save_epoch:
        with open(os.path.join(os.path.split(args.query_data_path)[0], 'aug_data_epoch_{}.pt'.format(data.epoch[0].item())), 'wb') as f:
            torch.save(data_list, os.path.join(os.path.split(args.query_data_path)[0], 'aug_data_epoch_{}.pt'.format(data.epoch[0].item())))
    
    loss_all /= total_graphs
    cls_loss_all /= total_graphs
    cl_loss_all /= total_graphs

    return loss_all, cls_loss_all, cl_loss_all


def train_node_weight_view_gen_and_cls_supervised(view_gen, view_optimizer, 
                                        model, optimizer, loader, device, args=None):
    view_gen.train()
    model.train()

    loss_all = 0
    cls_loss_all = 0
    total_graphs = 0
    
    for i, data in enumerate(loader):
        view_optimizer.zero_grad()
        optimizer.zero_grad()
        
        data = data.to(device)
        
        if args.encoder == 'ViewGenerator_subgraph_based_pipeline':
            view, _ = view_gen(data, True)
        elif args.encoder == 'ViewGenerator_subgraph_based_one':
            view, _ = view_gen(data, True)  
        elif args.encoder == 'ViewGenerator_based_one':
            view, _ = view_gen(data, True)
        else:
            raise ValueError("Unknown args.encoder %s".format(args.encoder))

        aug_output = model(view)
        raw_output = model(data)     
        
        aug_loss = F.nll_loss(aug_output, data.y)
        aug_acc = aug_output.max(1)[1].eq(data.y).sum().item() / data.num_graphs
        raw_loss = F.nll_loss(raw_output, data.y)
        raw_acc = raw_output.max(1)[1].eq(data.y).sum().item() / data.num_graphs
        cls_loss = aug_loss + raw_loss # classification loss
        
        loss = cls_loss
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs

        total_graphs += data.num_graphs
        view_optimizer.step()
        optimizer.step()
    
    loss_all /= total_graphs
    cls_loss_all /= total_graphs

    return loss_all, cls_loss_all



# train unlabel data 
def train_node_weight_view_gen_and_cls_unlabel(view_gen1, view_gen2, view_optimizer, 
                                        model, optimizer, loader, device, args=None):
    view_gen1.eval()
    view_gen2.eval()
    model.train()
    
    cl_loss_all = 0
    total_graphs = 0
    
    for i, data in enumerate(loader):
        view_optimizer.zero_grad()
        optimizer.zero_grad()
        
        data = data.to(device)

        if args.encoder == 'ViewGenerator_subgraph_based_pipeline':
            view1, _ = view_gen1(data, True)
            view2, _ = view_gen2(data, True)
        if args.encoder == 'ViewGenerator_subgraph_based_one':
            view1, _ = view_gen1(data, True)
            view2, _ = view_gen2(data, True)
        if args.encoder == 'ViewGenerator_based_one':
            view1, _ = view_gen1(data, True)
            view2, _ = view_gen2(data, True)
            
        output1 = model(view1)
        output2 = model(view2)        

        cl_loss = loss_cl(output1, output2)

        loss = cl_loss
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss.backward()

        cl_loss_all += cl_loss.item() * data.num_graphs

        total_graphs += data.num_graphs
        view_optimizer.step()
        optimizer.step()
    
    cl_loss_all /= total_graphs

    return cl_loss_all

@torch.no_grad()
def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    correct = 0
    num = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
        num = num + data.num_graphs
    return correct / num

@torch.no_grad()
def eval_result(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
    
    pred_list = []
    label_list = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1].cpu().detach().numpy().tolist()
            label = data.y.cpu().detach().numpy().tolist()
        pred_list += pred
        label_list += label
    f1_score_ = f1_score(label_list, pred_list, average='macro')
    accuracy_score_ = accuracy_score(label_list, pred_list)
    return f1_score_, accuracy_score_


@torch.no_grad()
def eval_acc_with_view_gen(view_gen1, view_gen2, model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
        view_gen1.eval()
        view_gen2.eval()

    correct = 0
    num = 0
    for data in loader:
        data = data.to(device)
        _, _, view1 = view_gen1(data)
        _, _, view2 = view_gen2(data)

        with torch.no_grad():
            pred1 = model(view1).max(1)[1]
            pred2 = model(view2).max(1)[1]

        correct1 = pred1.eq(data.y.view(-1)).sum().item()
        correct2 = pred2.eq(data.y.view(-1)).sum().item()
        num += data.num_graphs
        correct += (correct1 + correct2) / 2

    return correct / num


@torch.no_grad()
def eval_acc_with_node_weight_view_gen(view_gen1, view_gen2, model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
        view_gen1.eval()
        view_gen2.eval()

    correct = 0
    num = 0
    for data in loader:
        data = data.to(device)
        _, view1 = view_gen1(data, False)
        _, view2 = view_gen2(data, False)

        with torch.no_grad():
            pred1 = model(view1).max(1)[1]
            pred2 = model(view2).max(1)[1]

        correct1 = pred1.eq(data.y.view(-1)).sum().item()
        correct2 = pred2.eq(data.y.view(-1)).sum().item()
        num += data.num_graphs
        correct += (correct1 + correct2) / 2

    return correct / num

def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
    loss = 0
    num = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        num += data.num_graphs
    return loss / num

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_sum_exp(tensor, dim=None, keepdim=False):
    if dim is None:
        tensor_max = torch.max(tensor)
        return tensor_max + torch.log(torch.sum(torch.exp(tensor - tensor_max)))
    else:
        tensor_max, _ = torch.max(tensor, dim=dim, keepdim=True)
        result = tensor_max + torch.log(torch.sum(torch.exp(tensor - tensor_max), dim=dim, keepdim=keepdim))
        if not keepdim:
            result = result.squeeze(dim)
        return result