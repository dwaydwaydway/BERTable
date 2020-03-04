"""Downloads and prepares the Forest Covertype dataset."""
import argparse
import gzip
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import wget
import sys
from box import Box
from pathlib import Path

import ipdb


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
    if not os.path.exists(config.data_dir / 'covertype.csv'):
        os.makedirs(config.data_dir)

        filename = wget.download(config.url)
        with gzip.open(filename, 'rb') as f_in:
            with open(config.data_dir / 'covertype.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    df = pd.read_csv(config.data_dir / 'covertype.csv')
    n_total = len(df)

    indices = {}
    indices['train'], indices['test'] = train_test_split(
        range(n_total), test_size=config.test_size, random_state=config.random_state)
    indices['train'], indices['val'] = train_test_split(
        indices['train'], test_size=config.val_size, random_state=config.random_state)

    dataframe = {}
    for split in {'train', 'test', 'val'}:
        dataframe['split'] = df.iloc[indices['split']]
        if split == 'train':
            dataframe['split'] = dataframe['split'].sample(frac=1)
        dataframe['split'].to_csv(
            Path(config.data_dir) / f'{split}_covertype.csv', index=False, header=False)


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
