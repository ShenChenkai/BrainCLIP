import torch
import numpy as np
import random

def mixup_data(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    mixed_nodes = None
    if nodes is not None:
        mixed_nodes = lam * nodes + (1 - lam) * nodes[index, :]
        
    return mixed_x, mixed_nodes, y_a, y_b, lam


def mixup_data_by_class(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    mix_xs, mix_nodes, mix_ys = [], [], []

    for t_y in y.unique():
        idx = y == t_y

        t_mixed_x, t_mixed_nodes, _, _, _ = mixup_data(
            x[idx], nodes[idx] if nodes is not None else None, y[idx], alpha=alpha, device=device)
        mix_xs.append(t_mixed_x)
        if t_mixed_nodes is not None:
            mix_nodes.append(t_mixed_nodes)

        mix_ys.append(y[idx])

    if len(mix_nodes) > 0:
        return torch.cat(mix_xs, dim=0), torch.cat(mix_nodes, dim=0), torch.cat(mix_ys, dim=0)
    else:
        return torch.cat(mix_xs, dim=0), None, torch.cat(mix_ys, dim=0)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup(batch_data):
    x, edge_index, edge_attr, y, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.y, batch_data.batch
    bz = torch.max(batch) + 1
    x = x.reshape((bz, -1, x.shape[-1]))
    
    # Check if we can reshape edge_attr (dense case)
    # x.shape[1] is num_nodes per graph
    try:
        edge_attr_dense = edge_attr.reshape((bz, x.shape[1], -1))
        mixed_x, mixed_nodes, y_a, y_b, lam = mixup_data(x, edge_attr_dense, y)
        batch_data.edge_attr = edge_attr_dense.reshape((-1))
    except RuntimeError:
        # Sparse case or incompatible shape: don't mix edges
        mixed_x, _, y_a, y_b, lam = mixup_data(x, None, y)
        # Keep original edge_attr

    batch_data.x = mixed_x.reshape((-1, x.shape[-1]))

    return batch_data, y_a, y_b, lam



