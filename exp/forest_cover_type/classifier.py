import torch
import torch.nn as nn
import torch.nn.functional as F
# PositionalEmbedding credict to https://github.com/codertimo/BERT-pytorch

import ipdb


class Classifier(nn.Module):
    def __init__(
            self, bertable, embedding_dim, tab_len, dim_feedforward=100,
            n_layers=2, dropout=0.15, pooling='None'):
        super(Classifier, self).__init__()

        self.bertable = bertable
        in_dim = embedding_dim * tab_len if pooling == 'None' else embedding_dim

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Linear(in_dim, dim_feedforward))
            self.layers.append(nn.BatchNorm1d(dim_feedforward))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
            in_dim = dim_feedforward
        self.layers.append(
            nn.Linear(dim_feedforward, 7))

    def forward(self, batch_data):
        cls_o = self.bertable.forward(batch_data, mode='test')
        cls_o = cls_o.view(cls_o.size(0), -1)
        for layer in self.layers:
            cls_o = layer(cls_o)
        return cls_o
        
