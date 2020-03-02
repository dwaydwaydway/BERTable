import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# PositionalEmbedding credict to https://github.com/codertimo/BERT-pytorch

import ipdb


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Model(nn.Module):
    def __init__(
            self,
            vocab_size, col_type, use_pos,
            vector_dims, embedding_dim, dim_feedforward, tab_len,
            n_layers, n_head, dropout):
        super(Model, self).__init__()

        self.use_pos = use_pos

        if self.use_pos:
            self.position_enc = PositionalEmbedding(
                embedding_dim, sum(len(i) for i in col_type))

        self.exist_vector = len(vector_dims) > 0
        if self.exist_vector:
            self.vector_out = [nn.Linear(embedding_dim, dim)
                               for dim in vector_dims]

        self.embedding = nn.Embedding(
            vocab_size['numerical'] + vocab_size['categorical'] + 1, embedding_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu')

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm([tab_len, embedding_dim]))

        self.numerical_out = nn.Linear(embedding_dim, 1)
        self.categorical_out = nn.Linear(
            embedding_dim, vocab_size['categorical'])

        self.col_type = col_type
        self.embedding_dim = embedding_dim

        self.loss_f = {
            'categorical': nn.CrossEntropyLoss(),
            'vector': nn.MSELoss(reduction='mean')}

    def forward(
            self,
            batch,
            mode='test'):

        embeddding_o = self.embedding(
            batch['input']['idx']) * (batch['input']['weight'])

        if self.exist_vector:
            for i, col in enumerate(self.col_type['vector']):
                embeddding_o[:, col, :] = batch['input']['vector'][:, i, :]
        if self.use_pos:
            posi_enc = self.position_enc(batch['pos'])
            embeddding_o += posi_enc

        encoder_o = self.encoder(embeddding_o)

        if mode == 'train':
            logits, losses = {}, {}

            if 'numerical' in batch['gathering']:
                logits['numerical'] = self.numerical_out(
                    torch.masked_select(
                        encoder_o, batch['gathering']['numerical']).view(-1, self.embedding_dim))
                losses['numerical'] = ((logits['numerical'] - batch['labels']['numerical']) / (batch['std'] + 1e-8)) ** 2
                losses['numerical'] = torch.clamp(
                    losses['numerical'], 
                    self.loss_clip[0], 
                    self.loss_clip[1]).mean()

            if 'categorical' in batch['gathering']:
                logits['categorical'] = self.categorical_out(
                    torch.masked_select(
                        encoder_o, batch['gathering']['categorical']['encoder_o']).view(-1, self.embedding_dim))
                logits['categorical'] = torch.cat(
                    (logits['categorical'], batch['gathering']['categorical']['dummy_indices']), dim=1)
                logits['categorical'] = torch.gather(
                    logits['categorical'], 1, batch['gathering']['categorical']['samples'])
                losses['categorical'] = self.loss_f['categorical'](
                    logits['categorical'], batch['labels']['categorical'])

            if 'vector' in batch['gathering']:
                vector = torch.masked_select(
                    encoder_o, batch['gathering']['vector'])
                logits['vector'] = torch.stack(
                    [self.vector_out[i](vector[:, i, :]) for i in range(len(self.vector_out))])
                losses['vector'] = self.loss_f['vector'](
                    logits['vector'], batch['labels']['vector'])

            return encoder_o, losses

        return encoder_o
