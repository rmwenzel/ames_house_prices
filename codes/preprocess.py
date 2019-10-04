#!/usr/bin/env python3

from numpy import nan
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from statsmodels.imputation.mice import MICEData
from sklearn.model_selection import cross_val_score
from pandas import DataFrame
from math import isnan

import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore')


class DataDescription(dict):
    """Clean and convert dataset description text file into dict.

    Subclass of dict.

    Parameters
    ----------
    file_path: str
        Path to description text file

    """

    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            # build description dict
            lines = f.readlines()
            var_lines = [line for line in lines if ':' in line and
                         'one-half' not in line]

            desc_dict = DataDescription._build_desc_dict(lines, var_lines)
            desc_dict = DataDescription._clean_desc_dict(desc_dict)
            desc_dict = DataDescription._clean_var_names(desc_dict)

            # initialize self as dict and set
            dict.__init__(self, desc_dict)

    @staticmethod
    def _get_var_val_lines(lines, var_line, var_lines):
        """Get list of descriptions of values of variable."""
        var_val_lines = []

        # find file line corresponding to variable
        idx = lines.index(var_line) + 1
        line = lines[idx]

        # iteratively find descriptions for categorical variables
        while line not in var_lines:
            #  if line isn't only tab or newline character add it
            if set(line).difference({'\t', '\n'}):
                var_val_lines += [line]
            else:
                pass
            idx += 1
            try:
                line = lines[idx]
            except IndexError:
                break
        return var_val_lines

    @staticmethod
    def _clean_var_line(var_line):
        """Clean lines for non-categorical variables."""
        return var_line.replace('\t', '').replace('\n', '')

    @staticmethod
    def _clean_var_val_line(var_val_line):
        """Clean value lines for categorical variables."""
        var_val_line = var_val_line.split('\t')
        clean_var_line = []
        try:
            clean_var_line += [set(var_val_line[0].split(' ')).
                               difference({''}).pop()]
        except KeyError:
            clean_var_line += [None]
        try:
            clean_var_line += [var_val_line[1].replace('\n', '')]
        except IndexError:
            clean_var_line += [None]
        return clean_var_line

    @staticmethod
    def _build_desc_dict(lines, var_lines):
        """Build helper for setting desc_dict attribute."""
        desc_dict = {}
        for var_line in var_lines:

            clean_var_line = DataDescription._clean_var_line(var_line)
            var_val_lines = DataDescription._get_var_val_lines(lines,
                                                               var_line,
                                                               var_lines)

            clean_var_val_lines = []
            for var_val_line in var_val_lines:
                clean_var_val_lines += [DataDescription._clean_var_val_line(
                                        var_val_line)]

            desc_dict[clean_var_line] = {}
            for line in clean_var_val_lines:
                if line:
                    desc_dict[clean_var_line][line[0]] = line[1]
        return desc_dict

    @staticmethod
    def _clean_desc_dict(desc_dict):
        """Cleanup desc_dict."""
        clean_dict = {}
        for var_line in desc_dict:
            var_name, var_desc = var_line.split(':')
            var_values = desc_dict[var_line]

            clean_dict[var_name] = {'Description': var_desc}
            if var_values:
                clean_dict[var_name]['Values'] = var_values
        return clean_dict

    @staticmethod
    def _clean_var_names(desc_dict):
        """Correct specific variable names."""
        desc_dict['BedroomAbvGr'] = desc_dict['Bedroom']
        desc_dict.pop('Bedroom')
        desc_dict['KitchenAbvGr'] = desc_dict['Kitchen']
        desc_dict.pop('Kitchen')
        return desc_dict


class df_plus(DataFrame):
    """Augmented DataFrame class, subclass of pandas.DataFrame.

    Parameters
    ----------
    *args, **kwargs:
            Same as for pandas.DataFrame

    Attributes
    ----------
    col_kinds: dict
        Dictionary with keys ['cat', 'ord', 'quant'] and values a
        list of all variables in dataset of that type.
    """
    def __init__(self, *args, **kwargs):
        DataFrame.__init__(self, *args, **kwargs)
        self.col_kinds = {}
        self.LEs = {}

    # override superclass constructor to get cls instances from its methods
    @property
    def _constructor(self):
        return df_plus

    @property
    def col_kinds(self):
        """Getter for col_kinds attribute."""
        return self.col_kinds

    @col_kinds.setter
    def col_kinds(self, col_kinds):
        """Setter for col_kinds attribute.

        Parameters
        ----------
        col_kinds: dict
            Dictionary of column kinds. Keys must be one of
            'cat', 'ord', 'quant'.

        Raises
        ------
        AssertionError
            - Keys of col_kinds are not one of allowed keys
            - Values of col_kinds are not in list
            - Values of col_kinds are not strings
            - Self contains columns not in col_kinds

        """
        if isinstance(col_kinds, dict):
            allowed = {'cat', 'ord', 'quant'}
            try:
                for col_kind in col_kinds:
                    assert (col_kind in allowed)
            except AssertionError as e:
                e.args += ('Column type must be one of' +
                           str(allowed),)
                raise e
            try:
                for col_list in col_kinds.values():
                    assert (isinstance(col_list, list) or
                            isinstance(np.ndarray))
            except AssertionError as e:
                e.args += ('Column names must be in a list' +
                           ' or np.ndarray',)
                raise e
            try:
                for col_list in col_kinds.values():
                    for col_name in col_list:
                        assert (isinstance(col_name, str))
            except AssertionError as e:
                e.args += (str(col_name) + 'is not a string',)
                raise e
            try:
                cols_set = set()
                for col_kind in col_kinds.values():
                    cols_set = set(col_kind).union(cols_set)
                assert set(self.columns).issubset(cols_set)
            except AssertionError as e:
                e.args += ('df contains columns not in col_kinds',)
                raise e
            else:
                for col_kind in col_kinds:
                    curr_cols = set(self.columns)
                    kind_cols = set(col_kinds[col_kind])
                    self.col_kinds[col_kind] = list(curr_cols.
                                                    intersection(
                                                        kind_cols))
        else:
            raise ValueError('col_kinds must be a dict')

    def na_counts(self):
        """Get count of missing values.

        Returns
        -------
        na_counts: pandas.Series
            Null counts of columns with > 0 null values

        """
        na_counts = self.loc[:, self.isna().any()].isna().sum()
        return na_counts

    def na(self):
        """Get columns with missing values.

        Returns
        -------
        nan_df: pandas.DataFrame
            dataframe of all columns with missing values.

        """
        if self.na_counts().sum():
            nan_df = self.loc[:, self.isna().any()]
        else:
            nan_df = self
        return nan_df

    def null_cols_by_kind(self):
        """Group names of columns with null values by kind.

        Raises
        ------
        ValueError
            col_kinds attribute not set

        """
        if self.col_kinds:
            null_cols = {}
            for (col_kind, col_names) in self.col_kinds.items():
                null_names = list(set(self.na().columns).intersection(set(
                                  col_names)))
                null_cols[col_kind] = null_names
            return null_cols
        else:
            raise ValueError('col_kinds not set, use' +
                             'set_col_kinds')

    def encode_ords(self, mapper=None):
        """Numerically encode all values for ordinal columns.

        Parameters
        ----------
        mapper: dict, default is None
            Dictionary with keys old variables values and values new
            variable values. If mapper is None, all columns will be encoded
            with scikit-learn label encoders

        Raises
        ------
            ValueError
                Col_kinds attribute isn't set
            Exception


        Returns
        -------
        copy: pandas.Dataframe
            copy of self with ordinal column values encoded

        """
        copy = self.copy()

        # use mapper dictionary if passed
        if self.col_kinds and mapper:
            for col in mapper:
                copy.loc[:, col] = copy[col].map(lambda x:
                                                 mapper[col][x])
            self.LEs = {**self.LEs, **mapper}
            return copy

        # else use label encoders
        elif self.col_kinds and not mapper:
            # label encoders for all ordinal columns
            LEs = {col: LabelEncoder() for col
                   in self.col_kinds['ord']}
            for col in LEs:
                # fit the encoder to non null values
                LEs[col].fit(self[col].dropna())

                # transform non null values
                def f(x):
                    try:
                        return LEs[col].transform([x])[0]
                    except ValueError:
                        return nan

                copy.loc[:, col] = copy[col].map(f)

            self.LEs = {**self.LEs, **LEs}
            return copy

        else:
            raise ValueError('col_kinds not set, use' +
                             ' set_col_kinds')

    def impute_cat(self, col, *args, **kwargs):
        # Warning - needs work. Only use if ordinals are numerically encoded
        # encode other categoricals as dummy variables
        dummy_cols = list(set(self.col_kinds['cat']).difference({col}))
        df = pd.get_dummies(self, columns=dummy_cols).drop(
                            columns=['SalePrice'])

        df_not_na = df.dropna()

        # prepare non_null data for modeling
        X, y = df_not_na.drop(columns=[col]), df_not_na[col]
        X_sc = StandardScaler().fit_transform(X)

        # train model
        try:
            est = kwargs['estimator']
        except KeyError:
            est = XGBClassifier()
        finally:
            est.fit(X_sc, y)
        # impute predictions
        # fill any missing quantitative or ordinals with median value
        null_cols = self.null_cols_by_kind()
        quant_fill = {col: self.dropna()[col].median() for col in
                      null_cols['ord'] + null_cols['quant']}
        quant_fill.pop('SalePrice')
        X = df.fillna(value=quant_fill).drop(columns=[col])
        X_sc = StandardScaler().fit_transform(X)
        return est.predict(X_sc)

    def impute_cats(self, response, *args, **kwargs):
        null_cat_cols = self.null_cols_by_kind()['cat']
        try:
            null_cat_cols.remove(response)
        except ValueError:
            pass
        for col in null_cat_cols:
            self.loc[:, col] = self.impute_cat(col)

    def impute_quants(self, response, *args, **kwargs):
        # Warning - needs work. Only use if ordinals are numerically encoded
        cat_cols = self.col_kinds['cat']
        quant_cols = (self.col_kinds['quant'] + self.col_kinds['ord'])
        try:
            cat_cols.remove(response)
        except ValueError:
            pass
        try:
            quant_cols.remove(response)
        except ValueError:
            pass
        df = pd.get_dummies(self, columns=cat_cols)
        scaler = StandardScaler()
        df.loc[:, quant_cols] = scaler.fit_transform(df[quant_cols])
        df_imp = MICEData(df, perturbation_method='boot')
        df_imp.update_all(n_iter=6)
        self.loc[:, quant_cols] = scaler.inverse_transform(df_imp.data[
                                                           quant_cols])

    def impute(self, *args, **kwargs):
        # impute cats
        self.impute_cats()
        # impute quants
        self.impute_quants()

    def model(self, response, *args, **kwargs):
        resp_ser = self[response]
        df = pd.get_dummies(self, columns=self.col_kinds['cat'])
        sc = StandardScaler()
        df.loc['train'] = sc.fit_transform(df.loc['train'])
        df.loc['test'] = sc.transform(df.loc['test'])
        df.loc[:, response] = resp_ser
        return df


class hp_df_plus(df_plus):
    '''Subclass of df_plus for this dataset'''

    # override superclass constructor to get cls instances from its methods
    @property
    def _constructor(self):
        return hp_df_plus

    def __init__(self, *args, **kwargs):
        df_plus.__init__(self, *args, **kwargs)

    def print_desc(self, desc, cols=None):
        if cols is None:
            cols = self.columns
        # view description of all variables except sale price
        for col in cols:
            print(col + ':' + desc.dict_[col]['Description'] + '\n')
            try:
                for val in desc.dict_[col]['Values']:
                    print('\t', val + ' - ' + desc.dict_[col]['Values'][val])
            except KeyError:
                pass
            except TypeError:
                pass
            print('\n')

    def drop_prob_vars(self):
        df = self.copy()
        # drop columns with too many missing values
        notna_col_mask = ~ (df.isna().sum()/len(df) > 0.20)
        notna_col_mask.loc['SalePrice'] = True
        df = df.loc[:, notna_col_mask]
        df = df.drop(columns=['MiscVal'])

        # drop categorical variables with low number of classes and
        # extremely unbalanced distributions
        df = df.drop(columns=['Street', 'Utilities'])

        # drop outliers in OverallQual
        df = df.drop(df[(df['OverallQual'] < 5) &
                     (df['SalePrice'] > 200000)].index)

        # drop outliers in GrLivArea
        df = df.drop(df[(df['GrLivArea'] > 4000) &
                     (df['SalePrice'] < 300000)].index)
        self.drop_done = True
        return df

    def hand_impute(self):
        df = self.copy()
        # columns with null values by kind
        null_cols = self.null_cols_by_kind()

        cat_na_counts = df[null_cols['cat']].na_counts()
        ord_na_counts = df[null_cols['ord']].na_counts()
        quant_na_counts = df[null_cols['quant']].na_counts()

        # get col kinds with low number of missing values
        cat_lo_na_cols = cat_na_counts[cat_na_counts <= 4].index
        ord_lo_na_cols = ord_na_counts[ord_na_counts <= 2].index
        quant_lo_na_cols = quant_na_counts[quant_na_counts <= 1].index

        # fill categoricals with mode, quants with median
        missing_values = {**{col: df[col].mode().values[0] for col
                             in cat_lo_na_cols},
                          **{col: df[col].median() for col in
                             ord_lo_na_cols.union(quant_lo_na_cols)}}
        df = df.fillna(value=missing_values)
        return df

    def impute_quants(self, response, *args, **kwargs):
        # override method of superclass df_plus for this problem
        # Warning - needs work. Only use if ordinals are numerically encoded
        cat_cols = self.col_kinds['cat']
        quant_cols = (self.col_kinds['quant'] + self.col_kinds['ord'])
        try:
            cat_cols.remove(response)
        except ValueError:
            pass
        try:
            quant_cols.remove(response)
        except ValueError:
            pass
        df = pd.get_dummies(self, columns=cat_cols)
        scaler = StandardScaler()
        df.loc[:, quant_cols] = scaler.fit_transform(df[quant_cols])
        rename_cols = {'1stFlrSF': 'FirstFlrSF', '2ndFlrSF': 'SecondFlrSF',
                       '3SsnPorch': 'ThreeSsnPorch',
                       'HouseStyle_1.5Fin': 'HouseStyle_One_and_half_Fin',
                       'HouseStyle_1.5Unf': 'HouseStyle_One_and_half_Unf',
                       'HouseStyle_2.5Fin': 'HouseStyle_Two_and_half_Fin',
                       'HouseStyle_2.5Unf': 'HouseStyle_Two_and_half_Unf',
                       'Exterior2nd_Brk Cmn': 'Exterior2nd_BrkCmn',
                       'Exterior2nd_Wd Shng': 'Exterior2nd_WdShng',
                       'Exterior2nd_Wd Sdng': 'Exterior2nd_WdSdng',
                       'Exterior1st_Wd Sdng': 'Exterior1st_WdSdng',
                       'MSZoning_C (all)': 'MSZoning_C',
                       'RoofMatl_Tar&Grv': 'RoofMatl_TarGrv'
                       }
        df = df.rename(index=str, columns=rename_cols)
        df = df.drop(columns=[response])
        df_imp = MICEData(df, perturbation_method='boot')
        df_imp.update_all(n_iter=6)
        inv_rename_cols = {rename_cols[key]: key for key in rename_cols}
        df = df.rename(index=str, columns=inv_rename_cols)
        df_imp.data = df_imp.data.rename(index=str, columns=inv_rename_cols)
        df.loc[:, quant_cols] = scaler.inverse_transform(df_imp.data[
                                                         quant_cols])

        null_quant_cols = self.null_cols_by_kind()['quant']
        null_quant_cols += self.null_cols_by_kind()['ord']
        null_quant_cols.remove(response)
        self.loc[:,  null_quant_cols] = df[null_quant_cols].values


# collections of df_plus augmented DataFrames
class data_plus:

    def __init__(self, dfs):
        self.dfs = dfs

    def add_dfs(self, dfs, dfs_names=None):

        if isinstance(dfs, DataFrame):
            if not dfs_names:
                self.dfs[dfs_names] = dfs
            try:
                self.dfs[dfs_names] = dfs
            except TypeError as e:
                e.args += ('df_name was type' + str(type(dfs_names)))
                raise e.with_traceback(e.__traceback__)

        if isinstance(dfs, dict):
            try:
                for thing in dfs:
                    assert (isinstance(dfs[thing], DataFrame))
                    self.dfs[thing] = dfs[thing]
            except AssertionError as e:
                e.args += ('dfs should all be DataFrames',)
                raise e.with_traceback(e.__traceback__)
            except TypeError as e:
                raise e.with_traceback(e.__traceback__)

        else:
            try:
                for thing in dfs:
                    assert (isinstance(thing, DataFrame))
            except AssertionError as e:
                e.args += ('dfs should all be DataFrames',)
                raise e.with_traceback(e.__traceback__)

            try:
                assert(len(dfs) == len(dfs_names))
            except AssertionError as e:
                e.args += ('dfs and dfs_names must be the same length',)
                raise e.with_traceback(e.__traceback__)
            except TypeError as e:
                e.args += ('dfs and dfs_names should be iterables',)
                raise e.with_traceback(e.__traceback__)

            try:
                for name in dfs_names:
                    self.dfs[name] = dfs[name]
            except Exception as e:
                raise e

    def save_dfs(self):
        if self.dfs:
            try:
                for df_name in self.dfs:
                    self.dfs[df_name].droplevel(0).to_csv(df_name + '.csv', )
            except Exception as e:
                raise e


if __name__ == '__main__':

    start = time.time()
    print("Creating and processing original df 'orig'")
    train = pd.read_csv('train.csv', index_col='Id')
    test = pd.read_csv('test.csv', index_col='Id')
    full = pd.concat([train, test], keys=['train', 'test'], axis=0,
                     sort=False)
    orig = hp_df_plus(full)

    cat_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'Neighborhood',
                'LandContour', 'LotConfig', 'Condition1', 'Condition2',
                'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                'Heating', 'CentralAir', 'Electrical', 'GarageType',
                'MiscFeature', 'SaleType', 'SaleCondition']
    ord_cols = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'HeatingQC', 'BsmtFullBath', 'BsmtHalfBath',
                'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                'KitchenAbvGr', 'KitchenQual', 'Functional', 'Fireplaces',
                'FireplaceQu', 'GarageFinish', 'GarageCars', 'GarageQual',
                'GarageCond', 'LotShape', 'Fence', 'Utilities', 'LandSlope',
                'PavedDrive', 'PoolQC',  'MoSold', 'YrSold', 'YearBuilt',
                'YearRemodAdd', 'GarageYrBlt']
    quant_cols = ['GrLivArea',  'GarageArea', 'LotFrontage', 'LotArea',
                  'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                  'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                  'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                  'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']
    col_kinds = {'cat': cat_cols, 'ord': ord_cols, 'quant': quant_cols}
    # store all column kinds of original df
    orig.set_col_kinds(col_kinds)
    # start data_plus container
    hp_data = data_plus({'orig': orig})
    print("Processing of 'orig' complete:", time.time() - start, 'seconds')
    print()

    print("Creating and processing intermediate df 'edit':")
    start = time.time()
    # copy for editing
    edit = orig.copy()
    # set col kinds of copy
    edit = edit[cat_cols + ord_cols + quant_cols]
    edit.set_col_kinds(col_kinds)

    print('\tNumerically encoding ordinal variables')
    # mapper for ordinal columns values
    ords = {}
    ords['GarageCond'] = {nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    ords['BsmtCond'] = ords['GarageCond'].copy()
    ords['BsmtCond'].pop('Ex')
    ords['LandSlope'] = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
    ords['PavedDrive'] = {'N': 0, 'P': 1, 'Y': 2}
    ords['GarageFinish'] = {nan: 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    ords['BsmtQual'] = {nan: 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    ords['GarageQual'] = ords['GarageCond'].copy()
    ords['LotShape'] = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
    ords['Functional'] = {name: i for (i, name) in
                          enumerate(reversed(
                            orig['Functional'].unique()[:-1]))}
    ords['Functional'][nan] = nan
    ords['ExterCond'] = ords['GarageCond'].copy()
    ords['ExterQual'] = {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}
    ords['HeatingQC'] = ords['GarageCond'].copy()
    ords['KitchenQual'] = ords['BsmtQual'].copy()
    ords['BsmtFinType1'] = {nan: 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4,
                            'ALQ': 5, 'GLQ': 6}
    ords['BsmtFinType2'] = ords['BsmtFinType1'].copy()
    ords['BsmtExposure'] = {nan: 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    ords['Fence'] = {nan: 0, 'MnPrv': 1, 'MnWw': 2, 'GdWo': 3, 'GdPrv': 4}
    ords['FireplaceQu'] = ords['GarageCond'].copy()
    ords['PoolQC'] = ords['BsmtQual'].copy()
    # perform encoding
    edit = edit.encode_ords(mapper=ords)
    print('\tDropping problematic variables and two well known outliers')
    edit = edit.drop_prob_vars()
    # update col kinds
    edit.set_col_kinds(col_kinds)

    print('\tPerforming imputations:')
    print('\t\tImputing some missing values by hand')
    edit = edit.hand_impute()
    edit.set_col_kinds(col_kinds)

    print('\t\tImputing missing categorical values with XGBClassifier')
    edit.impute_cats(response='SalePrice')
    print('\t\tImputing missing quantitative values with MICE and PMM')
    edit.impute_quants(response='SalePrice')

    print('\t\tSetting dtypes')
    edit.set_col_kinds(col_kinds)
    cats, ords, quants = (edit.col_kinds['cat'], edit.col_kinds['ord'], edit.
                          col_kinds['quant'])
    edit.loc[:, cats] = edit.loc[:, cats].astype('category')
    edit.loc[:, ords] = edit.loc[:, ords].astype('int64')
    edit.loc[:, 'MSSubClass'] = edit['MSSubClass'].astype(str)
    edit.loc[:, 'MSSubClass'] = edit['MSSubClass'].astype('category')
    edit.loc[:, quants] = edit.loc[:, quants].astype('float64')
    print()

    print("Processing of 'edit' complete:", time.time() - start, 'seconds')
    print()

    print("Creating and processing model df 'model'")
    model = edit.model('SalePrice')
    model.loc[:, ords] = model.loc[:, ords].astype('int64')
    print()

    print("Storing dfs in container 'hp_data'")
    hp_data = data_plus({'orig': orig, 'edit': edit, 'model': model})
    print()

    print('Writing dfs to .csv files')
    for df_name in hp_data.dfs:
        df = hp_data.dfs[df_name].reset_index()
        df = df.rename({'level_0': 'split'}, axis='columns')
        df.to_csv(df_name + '.csv', index=False)
    print()

    print('Preprocessing complete. Total script runtime', time.time() - start,
          'seconds')
