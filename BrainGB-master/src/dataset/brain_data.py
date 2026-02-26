from torch_geometric.data import Data

class BrainData(Data):
    def __init__(self, num_views=None, **kwargs):
        super().__init__(**kwargs)
        self.num_views = num_views
        self.input_ids = kwargs.get('input_ids', None)
        self.attention_mask = kwargs.get('attention_mask', None)

    def __inc__(self, key, value, *args, **kwargs):
        if key.startswith('edge_index'):
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)
