import numpy as np
import pandas as pd
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
