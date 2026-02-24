from torch_geometric.data import Data

class BrainData(Data):
    def __init__(self, num_views=None, num_nodes=None, y=None, input_ids=None, attention_mask=None, **kwargs):
        # 1. 将标准的图属性 (y, num_nodes, edge_index, edge_attr等) 交给父类 Data 安全处理
        super(BrainData, self).__init__(y=y, num_nodes=num_nodes, **kwargs)

        # 2. 注册自定义属性 (PyG 会自动识别并在 DataLoader 中对它们进行 batch 拼接)
        self.num_views = num_views
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __inc__(self, key, value, *args, **kwargs):
        # 保持原有的图索引递增逻辑，保证多张图拼接成大图时，边索引正确偏移
        if key.startswith('edge_index'):
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)
