import xgboost as xgb
import pandas as pd
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
    for split in {'train', 'test', 'val'}:
        x[split] = pd.read_csv(config / f'{split}_covertype.csv', header=None)
        y[split] = x[split].iloc[len(x[split].columns)-1]
        x[split] = x[split].drop(len(x[split].columns)-1, axis=1)

    clf = xgb.XGBClassifier()

    param_grid = {
        'max_depth': list(range(config.param_grid.max_depth)),
        'learning_rate': list(range(config.param_grid.learning_rate)),
        'subsample': list(range(config.param_grid.subsample)),
        'colsample_bytree': list(range(config.param_grid.colsample_bytree)),
        'colsample_bylevel': list(range(config.param_grid.colsample_bylevel)),
        'colsample_bynode': list(range(config.param_grid.colsample_bynode)),
        'min_child_weight': list(range(config.param_grid.min_child_weight)),
        'gamma': list(range(config.param_grid.gamma)),
        'reg_lambda': list(range(config.param_grid.reg_lambda)),
        'n_estimators': list(range(config.param_grid.n_estimators))}

    fit_params = {
        'objective': config.fit_params.objective,
        'eval_metric': config.fit_params.eval_metric,
        'early_stopping_rounds': config.fit_params.early_stopping_rounds,
        'eval_set': [(x['val'], y['val'])]}

    rs_clf = GridSearchCV(
        clf, param_grid, 
        n_iter=20,
        n_jobs=config.num_workers, verbose=2, cv=2,
        fit_params=fit_params,
        scoring='accuracy', refit=False, random_state=config.random_state)

    print("Grid search..")
    search_time_start = time.time()
    rs_clf.fit(x['train'], y['train'])
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
