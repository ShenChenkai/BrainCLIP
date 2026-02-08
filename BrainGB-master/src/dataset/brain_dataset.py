import torch
import numpy as np
import os.path as osp

from scipy.io import loadmat
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from .base_transform import BaseTransform
from .brain_data import BrainData
import sys
from torch_geometric.data.makedirs import makedirs
from .abcd.load_abcd import load_data_abcd, load_data_abide, load_data_pnc
from torch_geometric.data.dataset import files_exist
import logging


def dense_to_ind_val(adj):
  
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = (torch.isnan(adj)==0).nonzero(as_tuple=True)
    edge_attr = adj[index]

    return torch.stack(index, dim=0), edge_attr


class BrainDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform: BaseTransform = None, view=0):
        self.view: int = view
        self.original_name = str(name)
        self.name_clean = self.original_name[:-4] if self.original_name.lower().endswith('.npy') else self.original_name
        self.name = self.name_clean.upper()
        self.filename_postfix = str(pre_transform) if pre_transform is not None else None
        self._builtin_names = ['PPMI', 'HIV', 'BP', 'ABCD', 'PNC', 'ABIDE']
        self.is_builtin = self.name in self._builtin_names
        super(BrainDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.num_nodes = torch.load(self.processed_paths[0])
        logging.info('Loaded dataset: {}'.format(self.name_clean))

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        if not self.is_builtin:
            return f'{self.name_clean}.npy'
        return f'{self.name}.mat'

    @property
    def processed_file_names(self):
        name = f'{self.name_clean}_{self.view}'
        if self.filename_postfix is not None:
            name += f'_{self.filename_postfix}'
        return f'{name}.pt'

    def _download(self):
        if files_exist(self.raw_paths) or self.name in ['ABCD', 'PNC', 'ABIDE'] or (not self.is_builtin):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        raise NotImplementedError

    def process(self):
        if not self.is_builtin:
            path = osp.join(self.raw_dir, self.raw_file_names)
            raw = np.load(path, allow_pickle=True)
            if isinstance(raw, np.ndarray) and raw.shape == ():
                raw = raw.item()
            if not isinstance(raw, dict):
                raise ValueError(
                    f'自定义数据集需要 .npy 保存为 dict，例如 {{"corr": corr, "label": label}}，但加载到的类型是 {type(raw)}')

            if 'corr' not in raw or 'label' not in raw:
                raise KeyError('自定义数据集 dict 必须包含键 "corr" 和 "label"')

            adj = np.asarray(raw['corr'])
            y = np.asarray(raw['label']).reshape(-1)

            y = torch.LongTensor(y)
            adj = torch.Tensor(adj)

            num_graphs = adj.shape[0]
            num_nodes = adj.shape[1]
        elif self.name in ['ABCD', 'PNC', 'ABIDE']:
            if self.name == 'ABCD':
                adj, y = load_data_abcd(self.raw_dir)
            elif self.name == 'PNC':
                adj, y = load_data_pnc(self.raw_dir)
            elif self.name == 'ABIDE':
                adj, y = load_data_abide(self.raw_dir)
            y = torch.LongTensor(y)
            adj = torch.Tensor(adj)
            num_graphs = adj.shape[0]
            num_nodes = adj.shape[1]
        else:
            m = loadmat(osp.join(self.raw_dir, self.raw_file_names))
            if self.name == 'PPMI':
                if self.view > 2 or self.view < 0:
                    raise ValueError(f'{self.name} only has 3 views')
                raw_data = m['X']
                num_graphs = raw_data.shape[0]
                num_nodes = raw_data[0][0].shape[0]
                a = np.zeros((num_graphs, num_nodes, num_nodes))
                for i, sample in enumerate(raw_data):
                    a[i, :, :] = sample[0][:, :, self.view]
                adj = torch.Tensor(a)
            else:
                key = 'fmri' if self.view == 1 else 'dti'
                adj = torch.Tensor(m[key]).transpose(0, 2)
                num_graphs = adj.shape[0]
                num_nodes = adj.shape[1]

            y = torch.Tensor(m['label']).long().flatten()
            y[y == -1] = 0

        data_list = []
        for i in range(num_graphs):
            edge_index, edge_attr = dense_to_ind_val(adj[i])
            data = Data(num_nodes=num_nodes, y=y[i], edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, num_nodes), self.processed_paths[0])

    def _process(self):
        print('Processing...', file=sys.stderr)

        if files_exist(self.processed_paths):  # pragma: no cover
            print('Done!', file=sys.stderr)
            return

        makedirs(self.processed_dir)
        self.process()

        print('Done!', file=sys.stderr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name_clean}()'
