import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import time
import json

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import (LinearRegression, Lasso, Ridge,
                                  RidgeCV, BayesianRidge)
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (KFold, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from functools import partial
from collections import defaultdict

# custom classes
from codes.process import DataDescription, HPDataFramePlus, DataPlus
from codes.explore import load_datasets


def print_small_val_counts(data, val_count_threshold):
    """Print counts of variables with any value count less than threshold."""
    small_val_count_columns = []
    for col in data.columns:
        val_counts = data[col].value_counts()
        if (val_counts < val_count_threshold).any():
            print()
            print(val_counts)
            small_val_count_columns += [col]
    return small_val_count_columns


def combine_MSSubClass(x):
    """Recode values of MSSubClass."""
    if x == 150:
        return 50
    else:
        return x


def combine_Condition2(x):
    """Recode values of Condition2."""
    if x in ['PosA', 'PosN', 'RRNn', 'RRAn', 'RRAe']:
        return 'Other'
    else:
        return x


def combine_RoofMatl(x):
    """Recode values of RoofMatl."""
    if x in ['WdShake', 'WdShingle', 'Wood']:
        return 'Wood'
    elif x in ['Roll', 'Metal', 'Membran']:
        return 'Other'
    else:
        return x


def combine_Exterior1st(x):
    """Recode values of Exterior1st."""
    if x == 'BrkComm':
        return 'BrkFace'
    elif x == 'AsphShn':
        return 'AsbShng'
    elif x == 'ImStucc':
        return 'Stucco'
    elif x in ['Stone', 'CBlock']:
        return 'Other'
    else:
        return x


def combine_Exterior2nd(x):
    """Recode values of Exterior2nd."""
    if x == 'AsphShn':
        return 'AsbShng'
    elif x in ['Stone', 'CBlock']:
        return 'Other'
    else:
        return x


def combine_Heating(x):
    """Recode values of Heating."""
    if x in ['Wall', 'OthW', 'Floor']:
        return 'Other'
    else:
        return x


def combine_MasVnrType(x):
    """Recode values of MasVnrType."""
    if x == 'BrkComm':
        return 'BrkFace'
    else:
        return x


def combine_Electrical(x):
    """Recode values of Electrical."""
    if x in ['FuseA', 'FuseF', 'FuseP', 'Mix']:
        return 'NonStd'
    else:
        return x


def combine_cat_vars(data):
    """Recode categorical variables."""
    copy = data.copy()
    combine_funcs = {}
    combine_funcs['MSSubClass'] = combine_MSSubClass
    combine_funcs['Condition2'] = combine_Condition2
    combine_funcs['RoofMatl'] = combine_RoofMatl
    combine_funcs['Exterior1st'] = combine_Exterior1st
    combine_funcs['Exterior2nd'] = combine_Exterior2nd
    combine_funcs['Heating'] = combine_Heating
    combine_funcs['MasVnrType'] = combine_MasVnrType
    combine_funcs['Electrical'] = combine_Electrical
    combine_funcs = {col: combine_funcs[col] for col in combine_funcs
                     if col in data.columns}
    for col in combine_funcs:
        copy.loc[:, col] = copy[col].apply(combine_funcs[col])
        copy.loc[:, col] = copy[col].astype('category')
    return copy


def create_ord_vars(edit_data, clean_data):
    """Engineer new features from ordinal variables.."""
    copy = edit_data.copy()

    # combine bathroom count variables and drop old ones
    copy['Bath'] = copy['HalfBath'] + 2 * copy['FullBath']
    copy['BsmtBath'] = copy['BsmtHalfBath'] + 2 * copy['BsmtFullBath']
    drop_cols = ['HalfBath', 'FullBath', 'BsmtHalfBath', 'BsmtFullBath']
    copy = copy.drop(columns=drop_cols)

    # create average quality and condition variables
    qual_cols, cond_cols = [], []
    for col in copy.columns:
        if 'Qu' in col or 'QC' in col:
            qual_cols += [col]
        elif 'Cond' in col and 'Sale' not in col:
            cond_cols += [col]
        else:
            pass
    copy['AvgQual'] = copy[qual_cols].mean(axis=1)
    copy['AvgCond'] = copy[cond_cols].mean(axis=1)

    # create indicator variables
    copy['HasBsmt'] = (clean_data['BsmtQual'] != 0).astype('int64')
    copy['HasFireplace'] = (clean_data['FireplaceQu'] != 0).astype('int64')
    copy['HasPool'] = (clean_data['PoolQC'] != 0).astype('int64')
    copy['HasGarage'] = (clean_data['GarageQual'] != 0).astype('int64')
    copy['HasFence'] = (clean_data['Fence'] != 0).astype('int64')

    return copy


def create_quant_vars(edit_data, clean_data):
    """Engineer new features from quantitative variables."""
    copy = edit_data.copy()

    # create indicator variables
    copy['Has2ndFlr'] = (clean_data['2ndFlrSF'] != 0).astype('int64')
    copy['HasWoodDeck'] = (clean_data['FireplaceQu'] != 0).astype('int64')
    copy['HasPorch'] = ((clean_data['OpenPorchSF'] != 0) |
                        (clean_data['EnclosedPorch'] != 0) |
                        (clean_data['3SsnPorch'] != 0) |
                        (clean_data['ScreenPorch'] != 0)).astype('int64')

    # create overall area variable
    copy['OverallArea'] = (copy['LotArea'] + copy['GrLivArea'] +
                           copy['GarageArea'])

    # create lot variable
    copy['LotRatio'] = copy['LotArea'] / copy['LotFrontage']

    return copy


def log_transform(data, log_cols):
    """Apply log transformation to quantitatives."""
    copy = data.copy()
    log_cols = [col for col in log_cols if col in data.columns]
    for col in log_cols:
        log_nonzero_min = np.log(copy[col][copy[col] != 0].min())
        copy['log_' + col] = copy[col].apply(lambda x: log_nonzero_min
                                             if x == 0 else np.log(x))
    copy = copy.drop(columns=log_cols)
    return copy


def do_col_kinds(hpdf):
    """Set col_kinds attribute."""
    col_kinds = {}
    col_kinds['cat'] = list(hpdf.data.select_dtypes('category').columns)
    col_kinds['ord'] = list(hpdf.data.select_dtypes('int64').columns)
    col_kinds['quant'] = list(hpdf.data.select_dtypes('float64').columns)
    hpdf.col_kinds = col_kinds


def train_test(data, response):
    """Train and test input and output data split."""
    X_train = data.loc['train'].drop(columns=[response])
    y_train = data.loc['train'][response]
    X_test = data.loc['train'].drop(columns=[response])
    y_test = data.loc['train'][response]
    return X_train, y_train, X_test, y_test


def build_model_data(hpdfs, data_names, response):
    """Compile all train and test input and output into dict."""
    model_data = defaultdict(dict)
    for (hpdf, data_name) in zip(hpdfs, data_names):
        X_train, y_train, X_test, y_test = train_test(hpdf.data, response)
        model_data[data_name]['X_' + data_name + '_train'] = X_train
        model_data[data_name]['y_' + data_name + '_train'] = y_train
        model_data[data_name]['X_' + data_name + '_test'] = X_test
        model_data[data_name]['y_' + data_name + '_test'] = y_test
    return model_data


def rmse(model, X_train, y_train, n_splits=5, shuffle=True,
         random_state=None):
    """Train and cv rmse of model."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    train_rmse = np.sqrt(mean_squared_error(model.predict(X_train), y_train))
    cv_scores = - cross_val_score(estimator=model, X=X_train, y=y_train,
                                  cv=kf, scoring='neg_mean_squared_error',
                                  n_jobs=-1)
    cv_rmse = np.sqrt(np.mean(cv_scores))
    return train_rmse, cv_rmse


def fit_default_models(model_data, random_state=None):
    """Fit some regressors to all datasets."""
    def_models = defaultdict(dict)
    for data_name in model_data:
        models = {'lasso': Lasso(), 'ridge': Ridge(),
                  'bayes_ridge': BayesianRidge(),
                  'plsreg': PLSRegression(), 'svr': SVR(),
                  'knn': KNeighborsRegressor(),
                  'mlp': MLPRegressor(),
                  'dec_tree': DecisionTreeRegressor(random_state=random_state
                                                    ),
                  'extra_tree': ExtraTreeRegressor(random_state=random_state),
                  'xgb': XGBRegressor(objective='reg:squarederror',
                                      random_state=random_state,
                                      n_jobs=-1)}
        for model in models:
            X_train = model_data[data_name]['X_' + data_name + '_train']
            y_train = model_data[data_name]['y_' + data_name + '_train']
            def_models[data_name][model] = models[model].fit(X_train, y_train)
    return def_models


def model_comparison(fit_models, data_name, model_data,
                     n_splits=5, shuffle=True, random_state=None):
    """Train and cv rmse for all models on version of data."""
    model_comp_df = pd.DataFrame(columns=['train_rmse', 'cv_rmse'],
                                 index=list(fit_models[data_name].keys()))
    for model_name in fit_models[data_name]:
        X_train = model_data[data_name]['X_' + data_name + '_train']
        y_train = model_data[data_name]['y_' + data_name + '_train']
        model = fit_models[data_name][model_name]
        train_rmse, cv_rmse = rmse(model, X_train, y_train, n_splits=10,
                                   shuffle=True, random_state=None)
        model_comp_df.loc[model_name, 'train_rmse'] = train_rmse
        model_comp_df.loc[model_name, 'cv_rmse'] = cv_rmse
    return model_comp_df


def compare_performance(fit_models, model_data, n_splits=5,
                        shuffle=True, random_state=None):
    """Compare default models across datasets."""
    comp_dfs = []
    for data_name in model_data:
        model_comp_df = model_comparison(fit_models, data_name, model_data,
                                         n_splits=5, shuffle=True,
                                         random_state=None)
        comp_dfs += [model_comp_df]
    comp_df = pd.concat(comp_dfs, axis=1, keys=list(model_data.keys()),
                        names=['data', 'performance'])
    comp_df = comp_df.reset_index().rename(columns={'index': 'model'})
    return comp_df


def plot_model_comp(comp_df, col, hue, **kwargs):
    """Plot results of model comparison."""
    df = comp_df.melt(id_vars='model')
    g = sns.catplot(x='model', y='value', data=df, col=col,
                    hue=hue, **kwargs)
    g.set_xticklabels(rotation=90)


def remove_models(models, drop_models):
    """Drop models from consideration."""
    copy = models.copy()
    for data_name in copy:
        for model in drop_models:
            copy[data_name].pop(model)
    return copy


def ho_cv_rmse(search_params, X_train, y_train, fixed_params={},
               est_name='bayes_ridge', n_splits=5, shuffle=True,
               random_state=None):
    """CV rmse objective function for hyperopt parameter search."""
    if est_name == 'ridge':
        est = Ridge(**{**search_params, **fixed_params})
    elif est_name == 'bridge':
        est = BayesianRidge(**{**search_params, **fixed_params})
    elif est_name == 'pls':
        est = PLSRegression(**{**search_params, **fixed_params})
    elif est_name == 'svr':
        est = SVR(**{**search_params, **fixed_params})
    elif est_name == 'xgb':
        est = XGBRegressor(**{**search_params, **fixed_params},
                           objective='reg:squarederror',
                           random_state=random_state, n_jobs=-1)
    est.fit(X_train, y_train)
    _, cv_rmse = rmse(model=est, X_train=X_train, y_train=y_train,
                      n_splits=n_splits, shuffle=shuffle,
                      random_state=random_state)
    return cv_rmse


def ho_results(obj, space, est_name, X_train, y_train,
               fixed_params={}, max_evals=10,
               n_splits=5, shuffle=True, random_state=None):
    """Hyperopt parameter search results."""
    fn = partial(obj, X_train=X_train, y_train=y_train,
                 fixed_params=fixed_params, est_name=est_name,
                 n_splits=n_splits, shuffle=shuffle,
                 random_state=random_state)
    trials = Trials()
    params = fmin(fn=fn, space=space, algo=tpe.suggest, trials=trials,
                  max_evals=max_evals)
    return trials, params


def rank_features(model, model_name, X_train, num_features,
                  figsize=None, rotation=None):
    """Plot most positive and negatively weighted features."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    coef_df = pd.DataFrame({'feature': X_train.columns, 'coef': model.coef_},
                           index=np.arange(X_train.shape[1]))

    plt.subplot(1, 2, 1)
    top_df = coef_df.sort_values(by='coef', ascending=False).head(num_features)
    plt.title('Best ' + model_name + ' ' + str(num_features) +
              ' most positive feature weights')
    sns.barplot(x='feature', y='coef', data=top_df, palette='Greens_r')
    plt.xticks(rotation=rotation)

    plt.subplot(1, 2, 2)
    bot_df = coef_df.sort_values(by='coef', ascending=False).tail(num_features)
    plt.title('Best ' + model_name + ' ' + str(num_features) +
              ' most negative feature weights')
    sns.barplot(x='feature', y='coef', data=bot_df, palette='Reds_r')
    plt.xticks(rotation=rotation)

    fig.tight_layout()


def compare_model_performance(fit_models, model_data, n_splits=5,
                              shuffle=True, random_state=None):
    """Compare train and cv rmse across datasets for single model."""
    comp_df = pd.DataFrame(columns=['train_rmse', 'cv_rmse'])
    for data_name in fit_models:
        X_train = model_data[data_name]['X_' + data_name + '_train']
        y_train = model_data[data_name]['y_' + data_name + '_train']
        for model_name in fit_models[data_name]:
            train_rmse, cv_rmse = rmse(fit_models[data_name][model_name],
                                       X_train, y_train,
                                       n_splits=n_splits, shuffle=shuffle,
                                       random_state=random_state)
            comp_df.loc[model_name, 'train_rmse'] = train_rmse
            comp_df.loc[model_name, 'cv_rmse'] = cv_rmse

    comp_df = comp_df.reset_index().rename(columns={'index': 'model'})
    return comp_df


def convert_to_int(ho_results, conv_params):
    """Workaround for hyperopt search not returning ints."""
    copy = ho_results.copy()
    for param in conv_params:
        copy = int(copy[param])


    return copy
