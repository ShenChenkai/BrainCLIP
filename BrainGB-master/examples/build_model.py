import logging
import torch
from src.models import GAT, GCN, BrainNN, MLP
from torch_geometric.data import Data
from typing import List


def build_model(args, device, model_name, num_features, num_nodes,
                load_pretrained=False, pretrained_path='pretrained_gcn.pth',
                freeze_gcn=False):
    if model_name == 'gcn':
        gcn = GCN(num_features, args, num_nodes, num_classes=2)

        if load_pretrained:
            state_dict = torch.load(pretrained_path, map_location=device)
            model_state = gcn.state_dict()
            # 过滤掉形状不匹配的权重，避免 edge_node_concate 等结构差异引发 RuntimeError
            compatible = {k: v for k, v in state_dict.items()
                          if k in model_state and v.shape == model_state[k].shape}
            skipped = [k for k in state_dict if k not in compatible]
            missing, unexpected = gcn.load_state_dict(compatible, strict=False)
            logging.info(f'Loaded pretrained GCN from {pretrained_path} '
                         f'({len(compatible)}/{len(state_dict)} layers transferred)')
            if skipped:
                logging.warning(f'  Skipped (shape mismatch): {skipped}')
            if missing:
                logging.warning(f'  Missing keys: {missing}')

            if freeze_gcn:
                for param in gcn.parameters():
                    param.requires_grad = False
                logging.info('GCN parameters frozen (Linear Probing mode)')

        model = BrainNN(args,
                      gcn,
                      MLP(2 * num_nodes, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),
                      ).to(device)
    elif model_name == 'gat':
        model = BrainNN(args,
                      GAT(num_features, args, num_nodes, num_classes=2),
                      MLP(2 * num_nodes, args.gat_hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),
                      ).to(device)
    else:
        raise ValueError(f"ERROR: Model variant \"{args.variant}\" not found!")
    return model
