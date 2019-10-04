#!/usr/bin/env python3

import pandas as pd
import scipy.stats as ss
import numpy as np

from itertools import combinations
from preprocess import data_plus, hp_df_plus


def chi_sq_ok(cont_tab_exp, min_freq, min_prop):
    cond1 = ((cont_tab_exp < min_freq).sum().sum() / cont_tab_exp.size <
             min_prop)
    cond2 = (cont_tab_exp == 0).sum().sum() == 0
    return cond1 and cond2


def chi_sq(df, min_freq, min_prop):
    # import pdb; pdb.set_trace()
    chi_sq_df = pd.DataFrame()
    for (col1, col2) in combinations(df.columns, 2):
        # cont_tab = sm.stats.Table.from_data(df[[col1, col2]])
        cont_tab = pd.crosstab(df[col1], df[col2])
        _, p_val, _, cont_tab_exp = ss.chi2_contingency(cont_tab)
        p_val = round(p_val, 5)
        if chi_sq_ok(cont_tab_exp, min_freq, min_prop):
            chi_sq_df.loc[col1, col1] = chi_sq_df.loc[col2, col2] = 0
            chi_sq_df.loc[col1, col2] = chi_sq_df.loc[col2, col1] = p_val
    return chi_sq_df


def cramers_v(df, chi_sq_df, alpha):
    cols = chi_sq_df.columns
    cramers_v_df = pd.DataFrame(columns=cols, index=cols)
    for (col1, col2) in combinations(cols, 2):
        import pdb; pdb.set_trace()
        if chi_sq_df.loc[col1, col2] < alpha:
            cont_tab = pd.crosstab(df[col1], df[col2])
            chi2, _, _, _ = ss.chi2_contingency(cont_tab)
            n = cont_tab.sum().sum()
            phi2 = chi2/n
            r, k = cont_tab.shape
            phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
            rcorr = r-((r-1)**2)/(n-1)
            kcorr = k-((k-1)**2)/(n-1)
            v = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
            cramers_v_df.loc[col1, col2] = cramers_v_df.loc[col2, col1] = v
            cramers_v_df.loc[col1, col1] = 1
    return cramers_v_df


if __name__ == '__main__':
    # load preprocessed data
    hp_data = data_plus({df_name: hp_df_plus(pd.read_csv(df_name + '.csv',
                                             index_col=['split', 'Id']))
                         for df_name in ['orig', 'edit', 'model']})
    cat_cols = ['MSSubClass', 'MSZoning', 'Street', 'LandContour',
                'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
                'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                'Heating', 'CentralAir', 'Electrical', 'GarageType',
                'MiscFeature', 'SaleType', 'SaleCondition', 'Alley']
    ord_cols = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual',
                'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual',
                'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'HeatingQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
                'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
                'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond',
                'PavedDrive', 'PoolQC', 'Fence']
    quant_cols = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
                  'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                  'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                  'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF',
                  'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                  'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']
    col_kinds = {'cat': cat_cols, 'ord': ord_cols, 'quant': quant_cols}
    for df_name in ['orig', 'edit']:
        hp_data.dfs[df_name].set_col_kinds(col_kinds)

    orig, edit, model = (hp_data.dfs['orig'], hp_data.dfs['edit'],
                         hp_data.dfs['model'])

    # analyze categorical variables
    cats = edit[edit.col_kinds['cat']]
    cats_chi_sq_df = chi_sq(cats, 5, 0.2)
    col_pairs = [(col1, col2) for (col1, col2) in
                 combinations(cats_chi_sq_df.columns, 2)
                 if cats_chi_sq_df.notna().loc[col1, col2]]
    cats_cramers_v_df = cramers_v(cats, cats_chi_sq_df, 0.01)
