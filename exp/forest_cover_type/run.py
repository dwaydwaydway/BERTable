import sys
sys.path.append('../..')
from exp.forest_cover_type.classifier import Classifier
from BERTable import BERTable
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import warnings
from pathlib import Path
from box import Box
import torch
import ipdb
import pandas as pd
import argparse
warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', dest='config_path',
        default='./config.yaml', type=Path,
        help='the path of config file')
    args = parser.parse_args()
    return vars(args)


class FitDataset(Dataset):
    def __init__(
            self, data, label, use_pos):

        self.data = data
        self.label = label

        self.use_pos = use_pos

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data['indices'])

    def __getitem__(self, index):
        sample = {'input': {}, 'labels': {}}

        idx, weight = self.data['indices'][index], self.data['weights'][index]

        sample['input']['idx'] = torch.LongTensor(idx)
        sample['input']['weight'] = torch.FloatTensor(weight)
        sample['label'] = self.label[index]

        if self.use_pos:
            sample['pos'] = torch.arange(0, len(idx))

        return sample

def collate_fn(batch):
    output = {'input': {}, 'gathering': {}, 'labels': {}}
    output['input']['idx'] = torch.stack(
        [data['input']['idx'] for data in batch])
    output['input']['weight'] = torch.stack(
        [data['input']['weight'] for data in batch]).unsqueeze(2)
    output['input']['weight'] = F.normalize(output['input']['weight'], dim=0)
    output['label'] = torch.LongTensor(
        [data['label'] for data in batch]
    )
    if 'pos' in batch[0]:
        output['pos'] = torch.stack([data['pos'] for data in batch])

    return output

def transfer(batch, device):
    batch['input']['idx'] = batch['input']['idx'].to(device)
    batch['input']['weight'] = batch['input']['weight'].to(device)
    batch['label'] = batch['label'].to(device)
    if 'pos' in batch:
        batch['pos'] = batch['pos'].to(device)
    return batch

def main(config_path):
    config = Box.from_yaml(config_path.open())

    df = pd.read_csv(
        Path(config.data_dir) / 'covertype.csv', header=None)

    column_type = ['numerical'] * 10 + ['categorical'] * 44
    if not config.use_label:
        df = df.drop(len(df.columns)-1, axis=1)
    df = df.values.tolist()

    bertable = BERTable(
        df, column_type,
        embedding_dim=config.model.embedding_dim,
        n_layers=config.model.n_layers,
        dim_feedforward=config.model.dim_feedforward,
        n_head=config.model.n_head,
        dropout=config.model.dropout,
        ns_exponent=config.model.ns_exponent,
        share_category=False, use_pos=False, device=device)
    if config.pretrain:
        bertable.pretrain(
            df,
            max_epochs=config.fit.max_epochs,
            lr=config.fit.lr,
            lr_weight={**config.fit.lr_weight},
            loss_clip=config.fit.loss_clip,
            n_sample=config.fit.n_sample,
            mask_rate=config.fit.mask_rate,
            replace_rate=config.fit.replace_rate,
            batch_size=config.fit.batch_size,
            shuffle=True,
            num_workers=config.num_workers)

    df, label = {}, {}
    for split in {'train', 'val'}:
        df[split] = pd.read_csv(
            Path(config.data_dir) / f'{split}_covertype.csv',
            header=None)
        label[split] = np.array(df[split][len(df[split].columns)-1].values.tolist()) - 1
        df[split] = df[split].drop(len(df[split].columns)-1, axis=1)
        df[split] = df[split].values.tolist()
        df[split] = bertable.vocab.convert(df[split], config.num_workers)

    clsf = Classifier(
        bertable.model,
        config.model.embedding_dim,
        len(column_type),
        dim_feedforward=config.cls.dim_feedforward,
        n_layers=config.cls.n_layers,
        dropout=config.cls.dropout,
        pooling=config.cls.pooling)

    optimizer = torch.optim.Adam(
        [{'params': clsf.bertable.parameters(), 'lr': 1e-4},
        {'params': clsf.layers.parameters()}], lr=float(config.cls.lr))
    clsf.to(device)

    process_bar = tqdm(
        range(config.cls.max_epochs),
        desc=f"[Progress]",
        total=config.cls.max_epochs,
        leave=True,
        position=0)

    loss_f = torch.nn.CrossEntropyLoss()

    for epoch in process_bar:
        for split in ['train', 'val']:
            generator = DataLoader(
                FitDataset(df[split], label[split], False),
                batch_size=config.fit.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                drop_last=False, 
                collate_fn=collate_fn)

            metric_bar = tqdm(
                [0],
                desc=f"[Metric]",
                bar_format="{desc} {postfix}",
                leave=False,
                position=2)

            epoch_bar = tqdm(
                generator,
                desc=f"[{split}]",
                leave=False,
                position=1)

            loss_history = []
            total, correct = 0, 0
            for batch_data in epoch_bar:

                batch_data = transfer(batch_data, device)
                logit = clsf(batch_data)
                loss = loss_f(logit, batch_data['label'])

                total += batch_data['label'].size(0)
                correct += (logit.argmax(dim=1) == batch_data['label']).sum().item()

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_history.append(loss.item())
                display = f'[{split}] Loss: {np.mean(loss_history):5.2f} | Acc: {correct / total:5.2f}'
                metric_bar.set_postfix_str(display)

            process_bar.write(f'[Log] Epoch {epoch:0>2d}| ' + display)
            epoch_bar.close()
            metric_bar.close()
    process_bar.close()

    # df = {}
    # for split in {'train', 'test', 'val'}:
    #     df[split] = pd.read_csv(
    #         Path(config.data_dir) / f'{split}_covertype.csv',
    #         header=None)
    #     label = df[split][len(df[split].columns)-1]
    #     label.to_csv(
    #         Path(config.data_dir) / f'{split}_bertable_y.csv',
    #         index=False, header=False)

    #     df[split] = df[split].drop(len(df[split].columns)-1, axis=1)
    #     df[split] = df[split].values.tolist()
    #     df_t = bertable.transform(
    #         df[split],
    #         batch_size=config.fit.batch_size,
    #         num_workers=config.num_workers)
    #     pd.DataFrame(df_t).to_csv(
    #         Path(config.data_dir) / f'{split}_bertable_x.csv',
    #         index=False, header=False)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
