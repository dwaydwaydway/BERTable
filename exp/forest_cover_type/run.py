import sys
sys.path.append('../..')
import argparse
import pandas as pd
import ipdb
import torch
import sys
from box import Box
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


from BERTable import BERTable
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
        config.data_dir / 'convertype.csv', header=None)

    column_type = ['numerical'] * 10 + ['categorical'] * 44
    df = df.values.tolist()
    
    bertable = BERTable(
        df, column_type,
        embedding_dim = config.model.bertable.embedding_dim, 
        n_layers = config.model.bertable.n_layers, 
        dim_feedforward = config.model.bertable.dim_feedforward, 
        n_head = config.model.bertable.n_head,
        dropout = config.model.bertable.dropout, 
        ns_exponent = config.model.bertable.ns_exponent, 
        share_category = False, use_pos = False, device = device)

    bertable.fit(
        df, 
        max_epochs = config.fit.max_epochs, 
        lr = config.fit.lr,
        lr_weight = {**config.fit.lr_weight},
        loss_clip = config.fit.loss_clip,
        n_sample = config.fit.n_sample, 
        mask_rate = config.fit.mask_rate, 
        replace_rate = config.fit.replace_rate, 
        batch_size = config.fit.batch_size, 
        shuffle = True, 
        num_workers = config.num_workers)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
