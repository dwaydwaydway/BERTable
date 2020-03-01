import xgboost as xgb

from sklearn.model_selection import GridSearchCV

x_train, y_train, x_valid, y_valid, x_test, y_test =  # load datasets

clf = xgb.XGBClassifier()

param_grid = {
    'max_depth': list(range(5, 50, 2)),
    'learning_rate': list(range(0, 1, 0.001)),
    'subsample': list(range(0, 1, 0.001)),
    'colsample_bytree': list(range(0, 1, 0.001)),
    'colsample_bylevel': list(range(0, 1, 0.001)),
    'colsample_bynode': list(range(0, 1, 0.001)),
    'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    'gamma': list(range(0.1, 5, 0.01)),
    'reg_lambda': list(range(0.1, 10, 0.5)),
    'n_estimators': list(range(50, 500, 5))}

fit_params = {
    'eval_metric': 'merror',
    'objective': 'multi:softmax',
    'early_stopping_rounds': 20,
    'eval_set': [(x_valid, y_valid)]}

rs_clf = GridSearchCV(clf, param_grid, n_iter=20,
                            n_jobs=10, verbose=2, cv=2,
                            fit_params=fit_params,
                            scoring='neg_log_loss', refit=False, random_state=42)
print("Grid search..")
search_time_start = time.time()
rs_clf.fit(x_train, y_train)
print("Randomized search time:", time.time() - search_time_start)

best_score = rs_clf.best_score_
best_params = rs_clf.best_params_
print("Best score: {}".format(best_score))
print("Best params: ")
for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))
