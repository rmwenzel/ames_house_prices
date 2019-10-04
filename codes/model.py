#!/usr/bin/env python3

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from functools import partial

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def xgb_reg_cv_rmse(hps, X, y):
    xgbreg = XGBRegressor(objective='reg:squarederror', **hps)
    return np.sqrt(-np.mean(cross_val_score(estimator=xgbreg,
                                            X=X,
                                            y=y,
                                            cv=10,
                                            scoring='neg_mean_squared_error',
                                            n_jobs=-1,
                                            )))


def main(full_edit, data, edit_cat_cols, edit_ord_cols, edit_quant_cols):
    print('Preparing data for modeling')
    # prepare data for modeling
    data = pd.get_dummies(full_edit, columns=edit_cat_cols)
    data['log_SalePrice'] = np.log(data['SalePrice'])
    cols = edit_ord_cols + edit_quant_cols
    data.loc[:, cols] = (data.loc[:, cols] -
                       data.loc[:, cols].max())/((
                          data.loc[:, cols].max() - data.loc[:, cols].min()))
    X = data.drop(columns=['SalePrice', 'log_SalePrice'])
    y = data['log_SalePrice']
    X_train = X.loc['train', :]
    X_test = X.loc['test', :]
    y_train = y.loc['train', :]

    print('Optimizing hyperparameters for XGBRegressor')
    hp_space = {
              'max_depth': ho_scope.int(hp.quniform('max_depth',
                                                    low=2, high=6, q=1)),
              'learning_rate': hp.loguniform('learning_rate',
                                             low=-4*np.log(10), high=0),
              'gamma': hp.loguniform('gamma',
                                     low=-3*np.log(10), high=2*np.log(10)),
              'n_estimators': ho_scope.int(hp.quniform('n_estimators',
                                                       low=100, high=500,
                                                       q=50))
              }
    best = fmin(fn=partial(xgb_reg_cv_rmse, X=X_train, y=y_train),
                space=hp_space, algo=tpe.suggest, max_evals=10)


if __name__ == '__main__':
    main()