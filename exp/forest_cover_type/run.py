import sys
sys.path.append('../..')
import argparse
import pandas as pd
import ipdb
import torch
from box import Box
from pathlib import Path
import warnings
import numpy as np
from BERTable import BERTable
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

    bertable.fit(
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
    for split in {'train', 'test', 'val'}:
        df[split] = pd.read_csv(
            Path(config.data_dir) / f'{split}_covertype.csv',
            header=None)
        label = df[split][len(df[split].columns)-1]
        label.to_csv(
            Path(config.data_dir) / f'{split}_bertable_y.csv',
            index=False, header=False)

        df[split] = df[split].drop(len(df[split].columns)-1, axis=1)
        df[split] = df[split].values.tolist()
        df_t = bertable.transform(
            df[split],
            batch_size=config.fit.batch_size,
            num_workers=config.num_workers)
        pd.DataFrame(df_t).to_csv(
            Path(config.data_dir) / f'{split}_bertable_x.csv',
            index=False, header=False)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
