#!/usr/bin/env python3

"""Helper functions for exploratory analysis of Ames housing dataset."""

import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import sys
import os
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
# custom classes
from codes.process import DataDescription, HPDataFramePlus, DataPlus

# add root site-packages directory to workaround pyitlib pip install issue
sys.path.append('/Users/home/anaconda3/lib/python3.7/site-packages')
from pyitlib import discrete_random_variable as drv

warnings.filterwarnings('ignore')
plt.style.use('seaborn-white')
sns.set_style('white')


def load_datasets(data_dir, file_names):
    """Load datasets by file name."""
    dfs = {}

    for file_name in file_names:
        path = os.path.join(data_dir, file_name)
        data = HPDataFramePlus.read_csv_with_dtypes(path, index_col=[0, 1])
        hpdf = HPDataFramePlus(data=data)
        df_name = file_name.split('.')[0]
        dfs[df_name] = hpdf
    hp_data = DataPlus(dfs)
    return hp_data


def plot_discrete_dists(nrows, ncols, data, figsize=None):
    """Plot distributions of discrete variables."""
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    for (i, col) in enumerate(data.columns):
        plt.subplot(nrows, ncols, i + 1)
        sns.countplot(data[col])
        plt.xticks(rotation=45, size='small')

    fig.tight_layout()
    plt.show()


def print_unbal_dists(data, bal_threshold):
    """Print distributions of columns highly concentrated as single value."""
    dists, unbal_cols = [], []
    for col in data.columns:
        val_counts = data[col].value_counts()
        dist = val_counts/sum(val_counts)
        if dist.max() > bal_threshold:
            dists += [dist]
            unbal_cols += [col]
    for dist in dists:
        print()
        print(dist)
    return unbal_cols


def print_val_counts(data, columns):
    """Print counts of values for columns."""
    for col in columns:
        print()
        print(data[col].value_counts())


def num_enc(data):
    """Numerically encode variables."""
    copy = data
    copy_num = copy.apply(LabelEncoder().fit_transform)
    return copy_num


def D(data, col1, col2, **kwargs):
    """Dependence distance for two dataframe columns."""
    var_info = drv.information_variation(data[[col1, col2]].T, **kwargs)
    joint_ent = drv.entropy_joint(data[[col1, col2]].T, **kwargs)
    D = (var_info/joint_ent)[0, 1]
    return D


def D_dep(data, **kwargs):
    """Depedence distance for pairs of variables."""
    dep_df = pd.DataFrame(columns=data.columns, index=data.columns)
    for (col1, col2) in combinations(data.columns, 2):
        dep_df.loc[col1, col2] = dep_df.loc[col2, col1] = D(data, col1, col2)
        dep_df.loc[col1, col1] = dep_df.loc[col2, col2] = 0.0
    for col in dep_df.columns:
        dep_df.loc[:, col] = pd.to_numeric(dep_df[col])
    return dep_df


def plot_D_dep(D_dep_df, figsize=None):
    """Plot heatmap of dependence distances."""
    plt.figure(figsize=figsize)
    plt.title('Dependence Distance')
    cmap = sns.cm.rocket_r
    # Generate a mask for the upper triangle
    mask = np.zeros_like(D_dep_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(D_dep_df, mask=mask, cmap=cmap)


def plot_low_D_dep(D_dep_df, D_threshold, figsize=None):
    """Plot heatmap of dependence distances less than threshold."""
    fig = plt.figure(figsize=figsize)
    plt.title('Low Dependence Distance')
    cmap = sns.cm.rocket_r
    # Generate a mask for the upper triangle
    mask = np.zeros_like(D_dep_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(D_dep_df[D_dep_df <= D_threshold],
                mask=mask, cmap=cmap, center=1, annot=True)
    fig.tight_layout()


def rank_pairs_by_D(D_dep_df, D_threshold):
    """Rank pairs of variables by dependence distance."""
    rank_D_df = pd.DataFrame(columns=['var1', 'var2', 'D'])
    for (col1, col2) in combinations(D_dep_df.columns, 2):
        D = D_dep_df.loc[col1, col2]
        rank_D_df = rank_D_df.append(pd.DataFrame([[col1, col2, D]],
                                     columns=rank_D_df.columns))
    rank_D_df = rank_D_df.reset_index(drop=True)
    rank_D_df = rank_D_df[rank_D_df['D'] <= D_threshold]
    rank_D_df = rank_D_df.sort_values('D', ascending=True)
    rank_D_df = rank_D_df.reset_index(drop=True)
    rank_D_df.index += 1
    return rank_D_df


def plot_violin_plots(nrows, ncols, data, response, figsize=None):
    """Plot distributions of response over discrete variable values."""
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    for (i, col) in enumerate(data.columns.drop(response)):
        plt.subplot(nrows, ncols, i + 1)
        sns.violinplot(x=col, y=response, data=data)
        plt.xticks(rotation=60)

    fig.tight_layout()


def D_dep_response(data, response, **kwargs):
    """Get dependence distance from response variable."""
    D_dep_resp_df = pd.DataFrame(index=data.drop(columns=[response]).columns,
                                 columns=['D'])
    for col in D_dep_resp_df.index:
        D_val = D(data, col, response, **kwargs)
        D_dep_resp_df.loc[col] = D_val
    return D_dep_resp_df


def rank_hyp_test(ords, stat_name, stat_function):
    """Wrap reporting results of rank correlation statistics."""
    cols = ords.data.drop(columns=['log_SalePrice']).columns
    stats = []
    for col in cols:
        stat = stat_function(ords.data.loc['train', :][col],
                             ords.data.loc['train', :]['log_SalePrice'])
        stats += [stat]
    stats_df = pd.DataFrame({stat_name: [l[0] for l in stats],
                             stat_name + '_p_val': [l[1] for l in stats]},
                            index=cols)
    return stats_df


def rank_by_col(df, col, ascending=True):
    """Rank rows by value in column."""
    sorted_df = df.sort_values(by=col, ascending=ascending)
    ranking = {idx: i + 1 for (i, idx) in enumerate(sorted_df.index)}
    df[col + '_rank'] = np.zeros(len(df))
    for idx in df.index:
        df.loc[idx, col + '_rank'] = ranking[idx]
    df[col + '_rank'] = df[col + '_rank'].astype('int64')
    return df


def get_rank_corr_df(rank_hyp_test_dfs):
    """Compare results of rank hypothesis tests."""
    dfs = []
    for df_name in rank_hyp_test_dfs:
        dfs += [rank_by_col(rank_hyp_test_dfs[df_name], df_name + '_p_val')]
    return pd.concat(dfs, axis=1)


def plot_cont_dists(nrows, ncols, data, figsize=None):
    """Plot distributions of continuous variables."""
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    for (i, col) in enumerate(data.columns):
        plt.subplot(nrows, ncols, i + 1)
        series = data[col][data[col].notna()]
        sns.distplot(series)
        plt.xticks(rotation=60)

    fig.tight_layout()


def plot_log_cont_dists(nrows, ncols, data, log_cols, figsize=None):
    """Plog log distributions of nonzero values of continuous variables."""
    log_df = data[log_cols]
    log_df = log_df[log_df > 0]
    log_df = np.log(log_df)
    plot_cont_dists(nrows, ncols, log_df, figsize=figsize)


def minmax_df(data):
    """Maximum and minimum values of all variables."""
    min_df = data.min()
    max_df = data.max()
    df = pd.concat([min_df, max_df], axis=1)
    df.columns = ['min', 'max']
    return df


def do_col_kinds(clean):
    """Set col_kinds attribute."""
    col_kinds = {}
    col_kinds['cat'] = list(clean.data.select_dtypes('category').columns)
    col_kinds['ord'] = list(clean.data.select_dtypes('int64').columns)
    col_kinds['quant'] = list(clean.data.select_dtypes('float64').columns)
    clean.col_kinds = col_kinds


def plot_corr(quants_data, figsize=None):
    """Plot (Pearson's) correlation as heatmap."""
    plt.figure(figsize=figsize)
    plt.title('Correlation/Linear Dependence')
    cmap = sns.cm.rocket_r
    corr = abs(quants_data.corr())
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, cmap=cmap)


def plot_high_corr(quants_data, abs_corr_threshold, figsize=None):
    """Plot (Pearson's) correlation with abs value less than threshold."""
    plt.figure(figsize=figsize)
    plt.title('Correlation/Linear Dependence')
    cmap = sns.cm.rocket_r
    corr = quants_data.corr()
    corr = corr[abs(corr) >= abs_corr_threshold]
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, cmap=cmap, center=1, annot=True)


def rank_pairs_by_abs_corr(quants_data, abs_corr_threshold):
    """Rank pairs of variables by abs value of Pearson's correlation."""
    rank_corr_df = pd.DataFrame(columns=['var1', 'var2', 'abs_corr'])
    for (col1, col2) in combinations(quants_data.columns, 2):
        abs_corr = abs(quants_data[[col1, col2]].corr().values[0, 1])
        df = pd.DataFrame([[col1, col2, abs_corr]],
                          columns=rank_corr_df.columns)
        rank_corr_df = rank_corr_df.append(df)
    rank_corr_df = rank_corr_df.reset_index(drop=True)
    rank_corr_df = rank_corr_df[rank_corr_df['abs_corr'] >=
                                abs_corr_threshold]
    rank_corr_df = rank_corr_df.sort_values('abs_corr').reset_index(drop=True)
    rank_corr_df.index += 1
    return rank_corr_df


def plot_joint_dists_with_response(nrows, ncols, quants_data, response,
                                   figsize=None):
    """Plot joint distributions of quantitatives and response."""
    columns = quants_data.columns.drop(response)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle('KDE estimate of joint distribution of quantitative' +
                 'variables and log of SalePrice', size=14)

    for (i, col) in enumerate(columns):
        j, k = i // ncols, i % ncols
        sns.kdeplot(data=quants_data.loc['train', :][col],
                    data2=quants_data.loc['train', :][response],
                    shade=True, shade_lowest=False, ax=axes[j, k])

    fig.tight_layout()
    fig.subplots_adjust(top=0.96)


def plot_scatter_with_response(nrows, ncols, quants_data, response):
    """Plot scatterplots of quantitatives and response."""
    columns = quants_data.columns.drop(response)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 30))
    fig.suptitle('Scatterplots of quantitative variables vs. ' +
                 'log of SalePrice', size=14)

    for (i, col) in enumerate(columns):
        j, k = i // ncols, i % ncols
        sns.scatterplot(x=quants_data.loc['train', :][col],
                        y=quants_data.loc['train', :][response],
                        data=quants_data, ax=axes[j, k])

    fig.tight_layout()
    fig.subplots_adjust(top=0.96)


def plot_log_scatter_with_response(nrows, ncols, quants_data, response):
    """Plot scatterplots of logs of quantitatives and response."""
    columns = quants_data.columns.drop(response)

    fig, axes = plt.subplots(nrows, ncols)
    fig.suptitle('Scatterplots of log of quantitative variables vs.'
                 + ' log of SalePrice', size=14)

    for (i, col) in enumerate(columns):
        j, k = i // ncols, i % ncols
        sns.scatterplot(x=np.log(quants_data.loc['train', :][col]),
                        y=quants_data.loc['train', :][response],
                        data=quants_data, ax=axes[j, k])
        axes[j, k].set_xlabel('log_' + col)

    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
