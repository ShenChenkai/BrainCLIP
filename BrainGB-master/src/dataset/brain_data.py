from torch_geometric.data import Data


class BrainData(Data):
    def __init__(self, num_views=None, num_nodes=None, y=None,
                 input_ids=None, attention_mask=None, *args, **kwargs):
        super(BrainData, self).__init__()
        self.num_views = num_views
        self.num_nodes = num_nodes
        self.y = y
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        for k, v in kwargs.items():
            if (k.startswith('x') or k.startswith('edge_index')
                    or k.startswith('edge_attr')
                    or k in ('input_ids', 'attention_mask')):
                self.__dict__[k] = v

    def __inc__(self, key, value):
        if key.startswith('edge_index'):
            return self.num_nodes
        else:
            return super().__inc__(key, value)
