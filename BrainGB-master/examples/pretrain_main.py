import argparse
import sys
import numpy as np
import torch
import os
import random
import logging

from src.dataset import BrainDataset
from src.utils import get_y
from src.models import GCN
from src.models.clip_brain import TextEncoder, BrainCLIP
from torch_geometric.loader import DataLoader
from .get_transform import get_transform
from .pretrain_and_evaluate import pretrain


def seed_everything(seed):
    print(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        loader_args = dict(num_workers=0, pin_memory=True)
    else:
        print("⚠️ Warning: CUDA is not available. Training will be slow on CPU.")
        loader_args = dict(num_workers=0)

    self_dir = os.path.dirname(os.path.realpath(__file__))

    if args.dataset_name == 'ABCD':
        root_dir = os.path.join(self_dir, 'datasets/ABCD/')
    elif args.dataset_name == 'PNC':
        root_dir = os.path.join(self_dir, 'datasets/PNC/')
    elif args.dataset_name == 'ABIDE':
        root_dir = os.path.join(self_dir, 'datasets/ABIDE/')
    else:
        root_dir = os.path.join(self_dir, 'datasets/')

    dataset = BrainDataset(root=root_dir,
                           name=args.dataset_name,
                           pre_transform=get_transform(args.node_features),
                           edge_sparsity=args.sparsity,
                           use_text=args.use_clip)

    num_features = dataset[0].x.shape[1]
    nodes_num = dataset.num_nodes

    # --- Determine GCN embedding dim based on pooling ---
    if args.pooling == 'concat':
        gcn_embedding_dim = 8 * nodes_num
    else:  # sum / mean
        gcn_embedding_dim = 256

    # --- Build BrainCLIP ---
    gcn_model = GCN(num_features, args, nodes_num, num_classes=2)
    text_encoder = TextEncoder(pretrained_model='prajjwal1/bert-tiny', proj_dim=256)
    model = BrainCLIP(gcn_model, text_encoder,
                      gcn_embedding_dim=gcn_embedding_dim, proj_dim=256).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Train on full dataset (no CV for pre-training) ---
    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, **loader_args)

    logging.info(f'Starting pre-training on {len(dataset)} samples ...')
    pretrain(model, train_loader, optimizer, device, args)

    # --- Save pre-trained GCN weights ---
    save_path = 'pretrained_gcn.pth'
    torch.save(model.gcn_model.state_dict(), save_path)
    logging.info(f'Pre-trained GCN saved to {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="FC_Y")
    parser.add_argument('--view', type=int, default=1)
    parser.add_argument('--sparsity', type=float, default=0.1)
    parser.add_argument('--node_features', type=str,
                        choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix',
                                 'eigenvector', 'eigen_norm'],
                        default='adj')
    parser.add_argument('--pooling', type=str,
                        choices=['sum', 'concat', 'mean'],
                        default='concat')

    parser.add_argument('--model_name', type=str, default='gcn')
    parser.add_argument('--gcn_mp_type', type=str, default="weighted_sum")
    parser.add_argument('--gat_mp_type', type=str, default="attention_weighted")

    parser.add_argument('--n_GNN_layers', type=int, default=2)
    parser.add_argument('--n_MLP_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--gat_hidden_dim', type=int, default=8)
    parser.add_argument('--edge_emb_dim', type=int, default=256)
    parser.add_argument('--bucket_sz', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_clip', action='store_true',
                        help='Enable text loading and CLIP pretraining/finetuning.')

    main(parser.parse_args())
