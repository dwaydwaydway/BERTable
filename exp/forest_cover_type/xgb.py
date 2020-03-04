import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
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

    x, y = {}, {}
    for split in {'train', 'val'}:
        x[split] = pd.read_csv(Path(config.data_dir) / f'{split}_bertable_x.csv', header=None).values
        y[split] = np.squeeze(pd.read_csv(Path(config.data_dir) / f'{split}_bertable_y.csv', header=None)).values

    # classifier
    clf = xgb.XGBClassifier(
        max_depth=20, n_estimators=400, learning_rate=0.03, n_jobs=config.num_workers, 
        subsample=0.5, colsample_bytree=0.7, colsample_bylevel=0.7, colsample_bynode=0.7, seed=4242, objective='multi:softmax')

    # fitting
    clf.fit(x['train'], y['train'], early_stopping_rounds=20,
    eval_metric="merror", eval_set=[(x['train'], y['train']), (x['val'], y['val'])])

    x, y = {}, {}
    x['test'] = pd.read_csv(Path(config.data_dir) / f'test_bertable_x.csv', header=None).values
    y['test'] = np.squeeze(pd.read_csv(Path(config.data_dir) / f'test_bertable_y.csv', header=None)).values

    preds = clf.predict(x['test'])
    
    print("Accuracy = {}".format(accuracy_score(y['test'], preds)))


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
