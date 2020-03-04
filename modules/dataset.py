import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ipdb


class PretrainDataset(Dataset):
    def __init__(
            self, data, col_type,
            vocab, embedding_dim, use_pos, mask_rate, replace_rate, n_sample):

        self.data = data
        self.col_type = col_type
        self.item2idx = vocab.item2idx
        self.neg_freq = vocab.neg_freq
        self.col_hash = vocab.col_hash
        self.vector_candidates = vocab.vector_candidates

        self.embedding_dim = embedding_dim
        self.n_sample = n_sample
        self.use_pos = use_pos
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.padding_index = 0

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data['indices'])

    def mask_idx(self, idx):
        replace_prob = np.random.rand(1).item()
        if replace_prob < self.replace_rate:
            return self.padding_index
        elif replace_prob > self.replace_rate + (1 - self.replace_rate)/2:
            return idx
        else:
            return torch.multinomial(self.neg_freq[idx], 1).item()

    def mask_weight(self, weight):
        return 0.0
        replace_prob = np.random.rand(1).item()
        if replace_prob < self.replace_rate:
            return 0.0
        elif replace_prob > self.replace_rate + (1 - self.replace_rate)/2:
            return weight
        else:
            return np.random.rand(1).item()

    def mask_vector(self, vector, col):
        replace_prob = np.random.rand(1).item()
        if replace_prob < self.replace_rate:
            return [0.0] * len(vector)
        elif replace_prob > self.replace_rate + (1 - self.replace_rate)/2:
            return vector
        else:
            return random.choice(self.vector_candidates[col])

    def __getitem__(self, index):
        sample = {'input': {}, 'labels': {}}

        idx, weight, fix_mask_prob = self.data['indices'][index], self.data[
            'weights'][index], self.data['fix_mask_prob'][index]

        sample['gathering'] = {}
        mask_prob = np.random.rand(len(idx)) + fix_mask_prob
        mask_prob = mask_prob < self.mask_rate

        sample['input']['idx'] = [self.mask_idx(
            idx[col]) if mask_prob[col] and col in self.col_type['categorical'] else idx[col] for col in range(len(idx))]
        sample['input']['weight'] = [self.mask_weight(
            weight[col]) if mask_prob[col] and col in self.col_type['numerical'] else weight[col] for col in range(len(idx))]

        if len(self.col_type['numerical']) > 0:
            sample['gathering']['numerical'] = [
                [mask_prob[col] and col in self.col_type['numerical']] * self.embedding_dim for col in range(len(idx))]
            sample['labels']['numerical'] = weight

        if len(self.col_type['categorical']) > 0:
            sample['gathering']['categorical'] = {
                'encoder_o': [[mask_prob[col] and col in self.col_type['categorical']] * self.embedding_dim
                                for col in range(len(idx))],
                'samples': []}
            label_idx = [idx[col] for col in self.col_type['categorical']
                            if mask_prob[col]]
            for value in label_idx:
                neg_sample = torch.multinomial(self.neg_freq[value], min(
                    len(torch.nonzero(self.neg_freq[value])), self.n_sample))
                if len(neg_sample) < self.n_sample:
                    neg_sample = torch.cat((neg_sample, torch.ones(
                        self.n_sample - len(neg_sample), dtype=torch.long) * len(self.item2idx['categorical'])))
                sample['gathering']['categorical']['samples'].append(
                    torch.cat((torch.LongTensor([value]), neg_sample)))
            sample['gathering']['categorical']['dummy_indices'] = [
                0] * len(label_idx)
            sample['labels']['categorical'] = [0] * len([
                col for col in self.col_type['categorical'] if mask_prob[col]])

        if len(self.col_type['vector']) > 0:
            vector = self.data['vector'][index]
            sample['input']['vector'] = [self.mask_vector(
                vector[col], col) if mask_prob[col] and col in self.col_type['vector'] else vector[col] for col in range(len(idx))]
            sample['gathering']['vector'] = [[mask_prob[col] <
                                                self.mask_rate and col in self.col_type['vector']] * self.embedding_dim for col in range(len(idx))]
            sample['labels']['vector'] = [vector[col]
                                            for col in self.col_type['vector'] if mask_prob[col]]

        if self.use_pos:
            sample['pos'] = torch.arange(0, len(idx))

        return sample

def collate_fn(batch):
    output = {'input': {}, 'gathering': {}, 'labels': {}}
    output['input']['idx'] = torch.LongTensor(
        [data['input']['idx'] for data in batch])
    output['input']['weight'] = torch.FloatTensor(
        [data['input']['weight'] for data in batch]).unsqueeze(2)
    output['input']['weight'] = F.normalize(output['input']['weight'], dim=0)

    if 'gathering' not in batch[0]:
        return output

    if 'numerical' in batch[0]['gathering']:
        output['gathering']['numerical'] = torch.ByteTensor(
            [data['gathering']['numerical'] for data in batch])
        output['labels']['numerical'] = torch.FloatTensor(
            [data['labels']['numerical'] for data in batch])
        output['std'] = torch.std(output['labels']['numerical'], dim=0,
                                  unbiased=True, keepdim=True).repeat(len(batch), 1)
        output['std'] = torch.masked_select(
            output['std'], output['gathering']['numerical'][:, :, 0]).unsqueeze(1)
        output['labels']['numerical'] = torch.masked_select(
            output['labels']['numerical'], output['gathering']['numerical'][:, :, 0]).unsqueeze(1)

    if 'categorical' in batch[0]['gathering']:
        output['gathering']['categorical'] = {}
        output['gathering']['categorical']['encoder_o'] = torch.ByteTensor(
            [data['gathering']['categorical']['encoder_o'] for data in batch])
        output['gathering']['categorical']['samples'] = torch.stack(
            [torch.LongTensor(samples) for data in batch for samples in data['gathering']['categorical']['samples']])
        output['gathering']['categorical']['dummy_indices'] = torch.FloatTensor(
            [idx for data in batch for idx in data['gathering']['categorical']['dummy_indices']]).view(-1, 1)
        output['labels']['categorical'] = torch.LongTensor(
            [label for data in batch for label in data['labels']['categorical']])

    if 'vector' in batch[0]['gathering']:
        output['input']['vector'] = torch.FloatTensor(
            [data['vector'] for data in batch])
        output['gathering']['vector'] = torch.ByteTensor(
            [data['gathering']['vector'] for data in batch])
        output['labels']['vector'] = [torch.FloatTensor(
            label) for data in batch for label in data['labels']['categorical']]

    if 'pos' in batch[0]:
        output['pos'] = torch.stack([data['pos'] for data in batch])

    return output


def transfer(batch, device):
    batch['input']['idx'] = batch['input']['idx'].to(device)
    batch['input']['weight'] = batch['input']['weight'].to(device)
    
    if 'gathering' in batch:
        if 'numerical' in batch['gathering']:
            batch['gathering']['numerical'] = batch['gathering']['numerical'].to(
                device)
            batch['labels']['numerical'] = batch['labels']['numerical'].to(device)
            batch['std'] = batch['std'].to(device)

        if 'categorical' in batch['gathering']:
            batch['gathering']['categorical']['encoder_o'] = batch['gathering']['categorical']['encoder_o'].to(
                device)
            batch['labels']['categorical'] = batch['labels']['categorical'].to(
                device)
            batch['gathering']['categorical']['samples'] = batch['gathering']['categorical']['samples'].to(
                device)
            batch['gathering']['categorical']['dummy_indices'] = batch['gathering']['categorical']['dummy_indices'].to(
                device)

        if 'vector' in batch['gathering']:
            batch['input']['vector'] = batch['input']['vector'].to(device)
            batch['gathering']['vector'] = batch['gathering']['vector'].to(
                device)
            batch['labels']['vector'] = batch['labels']['vector'].to(device)

    if 'pos' in batch:
        batch['pos'] = batch['pos'].to(device)

    return batch


def create_dataloader(
        data, col_type, vocab, embedding_dim, use_pos,
        batch_size, num_workers,
        mask_rate=0, replace_rate=0, n_sample=0, shuffle=False):

    dataset = PretrainDataset(
        data, col_type, vocab, embedding_dim, use_pos,
        mask_rate, replace_rate, n_sample)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn)

    return data_loader
