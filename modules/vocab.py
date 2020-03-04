import uuid
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

import ipdb

class Vocab():
    def __init__(
            self, 
            df, 
            col_type, 
            share_category, 
            ns_exponent):

        self.item2idx = {'numerical': {}, 'categorical': {}}
        self.neg_freq, self.vector_candidates, self.vetor_dim = {}, {}, {}
        col2candidate, idx2freq = {}, {}
        self.col_type = col_type

        if share_category:
            self.col_hash = [uuid.uuid4().hex[:6]] * sum([len(col_type[col])
                                                          for col in col_type])
        else:
            self.col_hash = set()
            while len(self.col_hash) < sum([len(col_type[col]) for col in col_type]):
                self.col_hash.add(uuid.uuid4().hex[:6])
            self.col_hash = list(self.col_hash)

        df_t = list(map(list, zip(*df)))
        for col, data in enumerate(tqdm(df_t, leave=False, desc=f"[Building Vocab]")):

            if col in col_type['numerical']:
                self.item2idx['numerical'][f'col_{col}'] = len(self.item2idx['numerical'])+1

            elif col in col_type['categorical']:
                category_count = Counter(data)
                category_count = {key:category_count[key] for key in category_count if key != np.nan}
                col2candidate[col] = [f'{self.col_hash[col]}{i}' for i in category_count]
                # assign id for each category
                for category in category_count:
                    hashed = f'{self.col_hash[col]}{category}'
                    if hashed not in self.item2idx['categorical']:
                        self.item2idx['categorical'][hashed] = len(self.item2idx['categorical'])+1
                        idx2freq[self.item2idx['categorical'][hashed]
                                 ] = category_count[category]
                    else:
                        idx2freq[self.item2idx['categorical'][hashed]
                                 ] += category_count[category]
            else:
                vectors = np.unique(np.array(data), axis=0)
                self.vetor_dim[col] = np.shape(vectors)[1]
                self.vector_candidates[col] = vectors

        for col in col_type['numerical']:
            self.item2idx['numerical'][f'col_{col}'] += len(self.item2idx['categorical'])

        # calculate sampling possibilities
        if share_category:
            total_count = sum([idx2freq[idx] for idx in idx2freq])
            for idx in idx2freq:
                idx2freq[idx] = idx2freq[idx] ** ns_exponent / total_count
                self.neg_freq[idx] = torch.zeros(
                    len(self.item2idx['categorical'])+1, dtype=torch.float)
                for other_idx in idx2freq:
                    if other_idx != idx:
                        self.neg_freq[idx][other_idx] = idx2freq[other_idx]
        else:
            for col in self.col_type['categorical']:
                total_count = sum([idx2freq[self.item2idx['categorical'][hashed]] for hashed in col2candidate[col]])
                for hashed in col2candidate[col]:
                    idx = self.item2idx['categorical'][hashed]
                    idx2freq[idx] = idx2freq[idx] ** ns_exponent / total_count
            for col in self.col_type['categorical']:
                for hashed in col2candidate[col]:
                    idx = self.item2idx['categorical'][hashed]
                    self.neg_freq[idx] = torch.zeros(
                        len(self.item2idx['categorical'])+1, dtype=torch.float)
                    for other_hashed in col2candidate[col]:
                        other_idx = self.item2idx['categorical'][other_hashed]
                        if other_idx != idx:
                            self.neg_freq[idx][other_idx] = idx2freq[other_idx]

    def convert(self, df, n_workers):
        self.df_t = list(map(list, zip(*df)))
        pool = Pool(processes=n_workers)
        result = pool.map(self.run_convert,
                          range(len(self.col_type['numerical']) + len(self.col_type['categorical'])))
        data = dict()
        data['indices'] = np.array([col[0]
                                    for col in result]).T
        data['weights'] = np.array([col[1]
                                   for col in result]).T
        data['fix_mask_prob'] = np.array([col[2]
                                          for col in result]).T
        data['vector'] = np.array([col[3]
                                   for col in self.col_type['vector']]).T
        del self.df_t
        return data

    def run_convert(self, col):
        if col in self.col_type['vector']:
            indices = [0] * len(self.df_t[col])
            weight = [0.0] * len(self.df_t[col])
            fix_mask_prob = [0] * len(self.df_t[col])
            return indices, weight, fix_mask_prob, self.df_t[col]
        elif col in self.col_type['numerical']:
            indices = [self.item2idx['numerical'][f'col_{col}']] * len(self.df_t[col])
            weight = [x if np.isfinite(x) else 0.0 for x in self.df_t[col]]
            fix_mask_prob = [0 if np.isfinite(
                x) else 1 for x in self.df_t[col]]
        elif col in self.col_type['categorical']:
            indices = [
                self.item2idx['categorical'][f'{self.col_hash[col]}{x}'] if f'{self.col_hash[col]}{x}' in self.item2idx['categorical'] else 0 for x in self.df_t[col]]
            weight = [
                1.0 if f'{self.col_hash[col]}{x}' in self.item2idx['categorical'] else 0.0 for x in self.df_t[col]]
            fix_mask_prob = [
                0 if f'{self.col_hash[col]}{x}' in self.item2idx['categorical'] else 1 for x in self.df_t[col]]
        return indices, weight, fix_mask_prob