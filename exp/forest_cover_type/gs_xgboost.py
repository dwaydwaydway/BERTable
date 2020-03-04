import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import time
from sklearn.model_selection import GridSearchCV
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
        x[split] = pd.read_csv(Path(config.data_dir) / f'{split}_bertable_x.csv', header=None)
        y[split] = pd.read_csv(Path(config.data_dir) / f'{split}_bertable_y.csv', header=None)

    clf = xgb.XGBClassifier()

    param_grid = {
        'max_depth': np.arange(*config.param_grid.max_depth),
        'learning_rate': np.arange(*config.param_grid.learning_rate),
        'subsample': np.arange(*config.param_grid.subsample),
        'colsample_bytree': np.arange(*config.param_grid.colsample_bytree),
        'colsample_bylevel': np.arange(*config.param_grid.colsample_bylevel),
        'colsample_bynode': np.arange(*config.param_grid.colsample_bynode),
        'min_child_weight': np.arange(*config.param_grid.min_child_weight),
        'gamma': np.arange(*config.param_grid.gamma),
        'reg_lambda': np.arange(*config.param_grid.reg_lambda),
        'n_estimators': np.arange(*config.param_grid.n_estimators)}

    fit_params = {
        'objective': config.fit_params.objective,
        'eval_metric': config.fit_params.eval_metric,
        'early_stopping_rounds': config.fit_params.early_stopping_rounds,
        'eval_set': [(x['train'], y['train']), (x['val'], y['val']), (x['test'], y['test'])]}

    rs_clf = GridSearchCV(
        clf, param_grid,
        n_jobs=config.num_workers, verbose=2, cv=5,
        scoring='accuracy', refit=False)

    print("Grid search..")
    search_time_start = time.time()
    rs_clf.fit(x['val'], y['val'], **fit_params)
    print("Grid search time:", time.time() - search_time_start)

    best_score = rs_clf.best_score_
    best_params = rs_clf.best_params_
    print("Best score: {}".format(best_score))
    print("Best params: ")
    for param_name in sorted(best_params.keys()):
        print('%s: %r' % (param_name, best_params[param_name]))


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        main(**args)
