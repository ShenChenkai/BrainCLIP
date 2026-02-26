import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model='prajjwal1/bert-tiny', proj_dim=256):
        super(TextEncoder, self).__init__()
        try:
            self.bert = AutoModel.from_pretrained(
                pretrained_model,
                use_safetensors=True,
                local_files_only=True,
            )
        except Exception:
            config = AutoConfig.from_pretrained(pretrained_model, local_files_only=True)
            self.bert = AutoModel.from_config(config)
        hidden_size = self.bert.config.hidden_size  # 128 for bert-tiny
        self.projection = nn.Linear(hidden_size, proj_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.projection(cls_embedding)


class BrainCLIP(nn.Module):
    def __init__(self, gcn_model, text_encoder, gcn_embedding_dim, proj_dim=256):
        super(BrainCLIP, self).__init__()
        self.gcn_model = gcn_model
        self.text_encoder = text_encoder
        self.graph_projection = nn.Linear(gcn_embedding_dim, proj_dim)

    def forward(self, x, edge_index, edge_attr, batch, input_ids, attention_mask):
        # Graph branch
        graph_features = self.gcn_model.forward_features(x, edge_index, edge_attr, batch)
        graph_proj = self.graph_projection(graph_features)

        # Text branch
        text_proj = self.text_encoder(input_ids, attention_mask)

        return graph_proj, text_proj
