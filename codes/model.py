"""Helpers for predictive modeling of SalePrice in Ames housing data in \
model.ipynb."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import copy
import os

from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from hyperopt import fmin, tpe, Trials
from functools import partial
from collections import defaultdict
from numpy import nan as NaN


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
    X_test = data.loc['test'].drop(columns=[response])
    y_test = data.loc['test'][response]
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

    y_pred = model.predict(X_train.values)
    train_rmse = np.sqrt(mean_squared_error(y_pred, y_train))
    cv_scores = - cross_val_score(estimator=model, X=X_train.values,
                                  y=y_train, cv=kf,
                                  scoring='neg_mean_squared_error',
                                  n_jobs=-1)
    cv_rmse = np.sqrt(np.mean(cv_scores))
    return train_rmse, cv_rmse


def fit_default_models(model_data, default_models):
    """Fit some regressors to all datasets."""
    fit_models = defaultdict(dict)
    for data_name in model_data:
        def_models = copy.deepcopy(default_models)
        for model in default_models:
            X_train = model_data[data_name]['X_' + data_name + '_train']
            y_train = model_data[data_name]['y_' + data_name + '_train']
            fit = def_models[model].fit(X_train.values, y_train.values)
            fit_models[data_name][model] = fit
    return fit_models


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


def rank_models_on_data(def_comp_cv, model_data):
    """Rank models by cv performance."""
    rank_df = def_comp_cv.copy()
    tups = list(zip(model_data.keys(), len(model_data)*['cv_rmse']))
    for tup in tups:
        df = def_comp_cv.sort_values(by=tup, ascending=True)
        for (i, model_name) in enumerate(df['model']):
            idx = def_comp_cv[def_comp_cv['model']
                              == model_name].index.values[0]
            rank_df.loc[idx, tup] = i + 1
    return rank_df


def rank_models_across_data(def_comp_cv, model_data):
    """Rank model cv across data sets."""
    rank_df = def_comp_cv.copy()
    for model_name in def_comp_cv['model']:
        df = def_comp_cv[def_comp_cv['model'] == model_name].T
        index = df.columns[-1]
        df = df.drop(index='model')
        df = df.sort_values(by=index)
        df = df.reset_index()
        for i in df.index:
            tup = (df.loc[i]['data'], df.loc[i]['performance'])
            model_idx = def_comp_cv[def_comp_cv['model']
                                    == model_name].index.values[0]
            rank_df.loc[model_idx, tup] = i + 1
    return rank_df


def remove_models(models, drop_models):
    """Drop models from consideration."""
    models = copy.deepcopy(models)
    for data_name in models:
        for model_name in drop_models:
            models[data_name].pop(model_name)
    return models


def ho_cv_rmse(search_params, X_train, y_train, est_name,
               fixed_params={}, n_splits=5, shuffle=True,
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
    est.fit(X_train.values, y_train.values)
    _, cv_rmse = rmse(model=est, X_train=X_train, y_train=y_train,
                      n_splits=n_splits, shuffle=shuffle,
                      random_state=random_state)
    return cv_rmse


def ho_results(obj, space, est_name, X_train, y_train,
               fixed_params={}, max_evals=10,
               n_splits=5, shuffle=True, random_state=None,
               trials=None):
    """Hyperopt parameter search results."""
    fn = partial(obj, X_train=X_train, y_train=y_train,
                 fixed_params=fixed_params, est_name=est_name,
                 n_splits=n_splits, shuffle=shuffle,
                 random_state=random_state)
    if not trials:
        trials = Trials()
    params = fmin(fn=fn, space=space, algo=tpe.suggest, trials=trials,
                  max_evals=max_evals)
    return {'trials': trials, 'params': params}


def plot_features(model, model_name, X_train, num_features,
                  figsize=None, rotation=None):
    """Plot most positive and negatively weighted features."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    coef_df = pd.DataFrame({'feature': X_train.columns, 'coef': model.coef_},
                           index=np.arange(X_train.shape[1]))

    plt.subplot(1, 2, 1)
    top_df = coef_df.sort_values(by='coef', ascending=False)
    top_df = top_df.head(num_features)
    plt.title('Best ' + model_name + ' ' + str(num_features) +
              ' most positive feature weights')
    sns.barplot(x='feature', y='coef', data=top_df, palette='Greens_r')
    plt.xticks(rotation=rotation)

    plt.subplot(1, 2, 2)
    bot_df = coef_df.sort_values(by='coef', ascending=False)
    bot_df = bot_df.tail(num_features)
    plt.title('Best ' + model_name + ' ' + str(num_features) +
              ' most negative feature weights')
    sns.barplot(x='feature', y='coef', data=bot_df, palette='Reds')
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
        copy[param] = int(copy[param])
    return copy


def plot_xgb_features(xgb_model, X_train, num_features, figsize=None,
                      rotation=None):
    """Plot most important features of xgb model."""
    plt.figure(figsize=figsize)
    imp_df = pd.DataFrame({'feature': X_train.columns,
                          'importance': xgb_model.feature_importances_},
                          index=np.arange(X_train.shape[1]))
    top_df = imp_df.sort_values(by='importance', ascending=False)
    top_df = top_df.head(num_features)
    plt.title('Best xgb model ' + str(num_features) +
              ' most important features')
    sns.barplot(x='feature', y='importance', data=top_df, palette='Greens_r')
    plt.xticks(rotation=rotation)


def pickle_to_file(models, file_path):
    """Pickle object to file."""
    with open(file_path, 'wb') as f:
        pickle.dump(models, f)


def pickle_from_file(file_path):
    """Load object from pickle file."""
    with open(file_path, 'rb') as f:
        models = pickle.load(f)
    return models


def convert_and_normalize_weights(voter_weights):
    """Normalize weight."""
    voter_weights = np.array(voter_weights)
    voter_weights /= voter_weights.sum()
    return voter_weights


def convert_search_params(search_params):
    """Package results of ensemble hyperparameter searchs into dict."""
    ridge_params = {key.replace('_ridge', ''): value for (key, value)
                    in search_params.items() if '_ridge' in key and '_meta'
                    not in key}
    svr_params = {key.replace('_svr', ''): value for (key, value)
                  in search_params.items() if '_svr' in key and
                  '_meta' not in key}
    xgb_params = {key.replace('_xgb', ''): value for (key, value)
                  in search_params.items()
                  if '_xgb' in key and '_meta' not in key}
    voter_weights = [value for (key, value) in search_params.items()
                     if 'voter' in key]
    meta_params = {key.replace('_meta', ''): value for (key, value)
                   in search_params.items() if '_meta' in key}
    params = defaultdict(dict)

    if ridge_params:
        params['ridge'] = ridge_params
    if svr_params:
        params['svr'] = svr_params
    if xgb_params:
        params['xgb'] = xgb_params
    if voter_weights:
        voter_weights = convert_and_normalize_weights(voter_weights)
        params['weights'] = voter_weights
    if meta_params:
        params['meta'] = convert_search_params(meta_params)

    return params


def voter_from_search_params(search_params, X_train, y_train,
                             random_state=None, fixed_params={}):
    """Fit a voting regressor with results of hyperparameter search."""
    voter_params = convert_search_params(search_params)

    conv_params = ['max_depth', 'min_child_weight', 'n_estimators']
    voter_params['xgb'] = convert_to_int(voter_params['xgb'], conv_params)
    voter_base_tuned = [('ridge', Ridge(**voter_params['ridge'])),
                        ('svr', SVR(**voter_params['svr'])),
                        ('xgb', XGBRegressor(**voter_params['xgb'],
                                             objective='reg:squarederror',
                                             n_jobs=-1,
                                             random_state=random_state))]
    weights_tuned = convert_and_normalize_weights(voter_params['weights'])
    voter = VotingRegressor(voter_base_tuned, weights=weights_tuned,
                            **fixed_params)
    voter.fit(X_train.values, y_train.values)
    return voter


def stack_from_search_params(search_params, X_train, y_train, meta_name,
                             random_state=None, fixed_params={}):
    """Fit a voting regressor with results of hyperparameter search."""
    stack_params = convert_search_params(search_params)

    conv_params = ['max_depth', 'min_child_weight', 'n_estimators']
    stack_params['xgb'] = convert_to_int(stack_params['xgb'], conv_params)

    try:
        xgb_params = stack_params['meta']['xgb']
        stack_params['meta']['xgb'] = convert_to_int(xgb_params, conv_params)
    except KeyError:
        pass

    base_ests = {'ridge': Ridge(**stack_params['ridge']),
                 'svr': SVR(**stack_params['svr']),
                 'xgb': XGBRegressor(**stack_params['xgb'],
                                     objective='reg:squarederror',
                                     n_jobs=-1, random_state=random_state)}
    meta_params = stack_params['meta'][meta_name]
    meta_est = copy.deepcopy(base_ests[meta_name]).set_params(**meta_params)
    stack = StackingCVRegressor(regressors=list(base_ests.values()),
                                meta_regressor=meta_est,
                                random_state=random_state,
                                **fixed_params)

    stack.fit(X_train.values, y_train.values)
    return stack


def ho_ens_cv_rmse(search_params, ens_name, X_train, y_train, pretuned=False,
                   base_ests=None, meta_ests=None, meta_name=None,
                   fixed_params={}, n_splits=5, shuffle=True,
                   random_state=None):
    """CV rmse objective for hyperopt parameter search for ensembles."""
    params = convert_search_params(search_params)
    if ens_name == 'voter' and pretuned:
        est = VotingRegressor(list(base_ests.items()),
                              weights=params['weights'], **fixed_params)
    elif ens_name == 'stack' and pretuned:
        est = StackingCVRegressor(regressors=list(base_ests.values()),
                                  meta_regressor=meta_ests[meta_name],
                                  **fixed_params)
    elif ens_name == 'voter':
        est = voter_from_search_params(search_params, X_train, y_train,
                                       fixed_params=fixed_params,
                                       random_state=random_state)

    elif ens_name == 'stack':
        est = stack_from_search_params(search_params, X_train, y_train,
                                       meta_name=meta_name,
                                       fixed_params=fixed_params,
                                       random_state=random_state)

    est.fit(X_train.values, y_train.values)
    _, cv_rmse = rmse(model=est, X_train=X_train, y_train=y_train,
                      n_splits=n_splits, shuffle=shuffle,
                      random_state=random_state)
    return cv_rmse


def ho_ens_results(obj, space, ens_name, X_train, y_train,
                   pretuned=False, base_ests=None,
                   meta_ests=None, meta_name=None, fixed_params={},
                   n_splits=5, shuffle=True,
                   random_state=None, max_evals=10, trials=None):
    """Hyperopt parameter search results for ensembles."""
    fn = partial(obj, ens_name=ens_name, X_train=X_train, y_train=y_train,
                 pretuned=pretuned, base_ests=base_ests, meta_ests=meta_ests,
                 meta_name=meta_name, fixed_params=fixed_params,
                 n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if not trials:
        trials = Trials()
    params = fmin(fn=fn, space=space, algo=tpe.suggest, trials=trials,
                  max_evals=max_evals)
    return {'trials': trials, 'params': params}


def add_stacks(ensembles, model_data, base_ests=None, meta_ests=None,
               suffix='', use_features_in_secondary=False,
               random_state=None):
    """Create stacking regressors from list of meta regressors."""
    ensembles = copy.deepcopy(ensembles)

    if not base_ests:
        ests = {'ridge': Ridge(),
                'svr': SVR(),
                'xgb': XGBRegressor(objective='reg:squarederror',
                                    n_jobs=-1,
                                    random_state=random_state)}
        base_ests = {data_name: ests for data_name in ensembles}

    if not meta_ests:
        ests = {'ridge': Ridge(),
                'svr': SVR(),
                'xgb': XGBRegressor(objective='reg:squarederror',
                                    n_jobs=-1,
                                    random_state=random_state)}
        meta_ests = {data_name: ests for data_name in ensembles}

    for data_name in ensembles:

        X_train = model_data[data_name]['X_' + data_name + '_train']
        y_train = model_data[data_name]['y_' + data_name + '_train']

        for meta_name in meta_ests[data_name]:
            stack_name = ('stack_' +
                          meta_name.replace('_tuned', '')
                          + '_' + suffix)
            if use_features_in_secondary:
                stack_name += '_second'

            bases = list(base_ests[data_name].values())
            meta = meta_ests[data_name][meta_name]
            stack = StackingCVRegressor(regressors=bases,
                                        meta_regressor=meta,
                                        use_features_in_secondary=use_features_in_secondary,
                                        random_state=random_state,
                                        n_jobs=-1)
            stack.fit(X_train.values, y_train.values)
            ensembles[data_name][stack_name] = stack

    return ensembles


def compare_ens_performance(ensembles, ens_ho_results, model_data,
                            n_splits=5, shuffle=True, random_state=None):
    """Compare default models across datasets."""
    comp_dfs = []
    for data_name in model_data:
        fit_models = defaultdict(dict)
        fit_models[data_name] = {model_name: model for (model_name, model)
                                 in ensembles[data_name].items() if
                                 model_name not in
                                 ens_ho_results[data_name]}
        model_comp_df = model_comparison(fit_models, data_name, model_data,
                                         n_splits=5, shuffle=True,
                                         random_state=None)
        for model_name in ens_ho_results[data_name]:
            trial = ens_ho_results[data_name][model_name][0]
            cv_rmse = trial.best_trial['result']['loss']
            X_train = model_data[data_name]['X_' + data_name + '_train']
            y_train = model_data[data_name]['y_' + data_name + '_train']
            y_pred = ensembles[data_name][model_name].predict(X_train.values)
            train_rmse = np.sqrt(mean_squared_error(y_pred, y_train))
            df = pd.DataFrame({'train_rmse': train_rmse, 'cv_rmse': cv_rmse},
                              index=[model_name])
            model_comp_df.append(df)
        comp_dfs += [model_comp_df]

    comp_df = pd.concat(comp_dfs, axis=1, keys=list(model_data.keys()),
                        names=['data', 'performance'])
    comp_df = comp_df.reset_index().rename(columns={'index': 'model'})

    return comp_df


def save_top_model_predictions(ensembles, ens_comp_df, data_name, model_data,
                               num_models, save_path):
    """Save top model predictions to .csv for submission to Kaggle."""
    top_df = ens_comp_df.sort_values(by=(data_name,
                                     'cv_rmse')).head(num_models)
    for model_name in top_df['model']:
        X_test = model_data[data_name]['X_' + data_name + '_test']
        y_pred = ensembles[data_name][model_name].predict(X_test.values)
        # transform predictions for log_SalePrice back to SalePrice
        y_pred = np.exp(y_pred)
        submit = pd.DataFrame({'Id': X_test.index, 'SalePrice': y_pred})
        file_name = model_name + '_' + data_name
        file_name = os.path.join(save_path, file_name)
        submit.to_csv(file_name, index=False)


def test_comp(ens_comp_df):
    """Empty DataFrame for comparison of final model test rmse."""
    ens_comp_ce = ens_comp_df.sort_values(by=('clean_edit', 'cv_rmse'))
    final_ce = ens_comp_ce.head(5)['model'].values
    final_ce = [name + '_clean_edit' for name in final_ce]
    ens_comp_de = ens_comp_df.sort_values(by=('drop_edit', 'cv_rmse'))
    final_de = ens_comp_de.head(5)['model'].values
    final_de = [name + '_drop_edit' for name in final_de]
    final_model_names = final_ce + final_de
    return pd.DataFrame({'models': final_model_names, 'test_rmse': NaN})
