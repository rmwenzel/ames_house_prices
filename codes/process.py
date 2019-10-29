#!/usr/bin/env python3

"""Script for preprocessing Ames housing dataset."""

from numpy import nan
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
from statsmodels.imputation.mice import MICEData

import numpy as np
import pandas as pd
import warnings
import time
import os
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
        """Class constructor."""
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


class DataFramePlus:
    """Augmented pandas DataFrame class.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame containing data
    col_kinds: dict, default empty
        Dictionary with keys ['cat', 'ord', 'quant'] and values a
        list of all variables in dataset of that type.
    label_encoders: dict, default empty
        Dictionary with keys column names and values sckit learn label encoders
        fit on values from corresponding columns
    *args, **kwargs:
            Same as for pandas.DataFrame

    Attributes
    ----------
    data: pandas.DataFrame
        DataFrame containing data
    col_kinds: dict
        Dictionary with keys ['cat', 'ord', 'quant'] and values a
        list of all variables in dataset of that type.
    label_encoders: dict
        Dictionary with keys column names and values sckit learn label
        encoders fit on values from corresponding columns

    """

    def __init__(self, data, col_kinds={}, label_encoders={},
                 *args, **kwargs):
        """Class constructor."""
        self._data = pd.DataFrame(data, *args, **kwargs)
        self._col_kinds = col_kinds
        self._label_encoders = label_encoders

    @property
    def data(self):
        """Get data attribute."""
        return self._data

    @data.setter
    def data(self, data):
        """Set data attribute."""
        self._data = data

    @data.deleter
    def data(self):
        """Delete data attribute."""
        del self._data

    @property
    def col_kinds(self):
        """Get col_kinds attribute."""
        return self._col_kinds

    @col_kinds.setter
    def col_kinds(self, col_kinds):
        """Set col_kinds attribute."""
        self._col_kinds = col_kinds

    @col_kinds.deleter
    def col_kinds(self):
        """Delete col_kinds attribute."""
        del self._col_kinds

    @property
    def label_encoders(self):
        """Get label_encoders attribute."""
        return self._label_encoders

    @label_encoders.setter
    def label_encoders(self, label_encoders):
        """Set label_encoders attribute."""
        self._label_encoders = label_encoders

    @label_encoders.deleter
    def label_encoders(self):
        """Delete label_encoders attribute."""
        del self._label_encoders

    def update_col_kinds(self, col_kinds):
        """Update col_kinds attribute.

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

            # check that col_kinds keys are allowed
            try:
                for col_kind in col_kinds:
                    assert (col_kind in allowed)
            except AssertionError as e:
                e.args += ('Column type must be one of ' +
                           str(allowed),)
                raise e

            # check that col_kinds values are array-like
            try:
                for col_list in col_kinds.values():
                    assert (isinstance(col_list, list) or
                            isinstance(np.ndarray))
            except AssertionError as e:
                e.args += ('Column names must be in a list' +
                           ' or np.ndarray',)
                raise e

            # check entries in col_kinds values are strings
            try:
                for col_list in col_kinds.values():
                    for col_name in col_list:
                        assert (isinstance(col_name, str))
            except AssertionError as e:
                e.args += (str(col_name) + 'is not a string',)
                raise e

            # check that dataframe columns are spoken for
            try:
                cols_set = set()
                for col_kind in col_kinds.values():
                    cols_set = set(col_kind).union(cols_set)
                assert set(self.data.columns).issubset(cols_set)
            except AssertionError as e:
                e.args += ('df contains columns not in col_kinds',)
                raise e

            # update col_kinds attribute
            else:
                new_col_kinds = {}
                for col_kind in col_kinds:
                    curr_cols = set(self.data.columns)
                    kind_cols = set(col_kinds[col_kind])
                    new_col_kinds[col_kind] = list(curr_cols.
                                                   intersection(
                                                        kind_cols))
                self.col_kinds = new_col_kinds
        else:
            raise ValueError('col_kinds must be a dict')

    def na_counts(self):
        """Get count of missing values.

        Returns
        -------
        na_counts: pandas.Series
            Counts of missing values for columns with > 0 missing values

        """
        na_counts = self.data.loc[:, self.data.isna().any()].isna().sum()
        return na_counts

    def na(self):
        """Get columns with missing values.

        Returns
        -------
        nan_df: DataFramePlus
            dataframe of all columns with missing values.

        """
        if self.na_counts().sum():
            nan_df = self.data.loc[:, self.data.isna().any()]
        else:
            nan_df = self
        return nan_df

    def na_cols_by_kind(self):
        """Group names of columns with missing values by kind.

        Raises
        ------
        ValueError
            col_kinds attribute is empty


        Returns
        -------
        na_cols: dict
            dictionary with keys column kinds ('cat', 'ord', 'quant') and
            values lists of names of columns of that kind with missing values

        """
        if self.col_kinds:
            na_cols = {}
            for (col_kind, col_names) in self.col_kinds.items():
                na_names = list(set(self.na().columns).intersection(set(
                                  col_names)))
                na_cols[col_kind] = na_names
            return na_cols
        else:
            raise ValueError('col_kinds is empty')

    def encode_ords(self, mapper={}):
        """Numerically encode values for all ordinal columns.

        Parameters
        ----------
        mapper: dict, default empty
            Dictionary with keys old columns values and values new
            variable values. If mapper is None, all columns will be encoded
            with scikit-learn label encoders

        Raises
        ------
            ValueError
                Col_kinds attribute is empty
            Exception


        Returns
        -------
        copy: DataFramePlus
            copy of self with ordinal column values encoded

        """
        # update column kinds
        self.update_col_kinds(self.col_kinds)

        copy = self.data.copy()

        # use mapper dictionary if passed
        if self.col_kinds and mapper:
            for col in mapper:
                copy.loc[:, col] = copy[col].map(lambda x:
                                                 mapper[col][x])
            self.label_encoders = {**self.label_encoders, **mapper}
            return copy

        # else use label encoders
        elif self.col_kinds and not mapper:
            # label encoders for all ordinal columns
            label_encoders = {col: LabelEncoder() for col
                              in self.col_kinds['ord']}
            for col in label_encoders:
                # fit the encoder to non null values
                label_encoders[col].fit(self[col].dropna())

                # transform non null values
                def trans(x):
                    try:
                        return label_encoders[col].transform([x])[0]
                    except ValueError:
                        return nan

                copy.loc[:, col] = copy[col].map(trans)

            self.label_encoders = {**self.label_encoders, **label_encoders}
            return copy

        else:
            raise ValueError('col_kinds not set, use' +
                             ' set_col_kinds')

    def impute_cat(self, col, response, classifier=None, *args, **kwargs):
        """Impute missing values of categorical column.

        Assumes ordinal columns are already numerically encoded and imputed.

        Parameters
        ----------
        col: str
            Name of column to be imputed
        response, str
            Name of response variable column
        classifier: scikit-learn classifier, default None
            Scikit learn classifier to be used for prediction.
            If None, XGBoost classifier is used.

        Returns
        -------
        imputed: numpy.ndarray
            numpy array of imputed values

        """
        # one-hot encode other columns
        dummy_cols = list(set(self.col_kinds['cat']).difference({col}))
        df = pd.get_dummies(self.data, columns=dummy_cols).drop(
                            columns=[response])

        df_not_na = df.dropna()

        # prepare non-null data for prediction
        X, y = df_not_na.drop(columns=[col]), df_not_na[col]
        X_sc = MinMaxScaler().fit_transform(X)

        # fit classifier
        if not classifier:
            classifier = XGBClassifier()
        classifier.fit(X_sc, y)

        # fill any missing quantitative or ordinals with median value
        na_cols = self.na_cols_by_kind()
        quant_fill = {col: self.data.dropna()[col].median() for col in
                      na_cols['ord'] + na_cols['quant']}
        quant_fill.pop('SalePrice')
        X = df.fillna(value=quant_fill).drop(columns=[col])
        X_sc = MinMaxScaler().fit_transform(X)

        # predict categorical column
        imputed = classifier.predict(X_sc)
        return imputed

    def impute_cats(self, response, classifier=None, *args, **kwargs):
        """Impute all missing categorical column values using classifier.

        Assumes ordinal columns are already numerically encoded and imputed.

        Parameters
        ----------
        response: str
            Column name for response variable. Missing values in this column
            will not be imputed
        classifier: scikit-learn classifier, default None
            Scikit learn classifier to be used for prediction.
            If None, XGBoost classifier is used.

        Returns
        -------
        copy: DataFramePlus
            copy of self with categorical column values imputed

        """
        # update column kinds
        self.update_col_kinds(self.col_kinds)

        copy = self.data.copy()

        # get names of categorical columns with missing values
        na_cat_cols = self.na_cols_by_kind()['cat']

        # remove response column if need be
        try:
            na_cat_cols.remove(response)
        except ValueError:
            pass

        # iteratively impute missing values
        for col in na_cat_cols:
            copy.loc[:, col] = self.impute_cat(col, response, classifier)

        return copy

    def impute_quants(self, response, n_iter=1, *args, **kwargs):
        """Impute all missing quantitative column values using MICE.

        See https://www.statsmodels.org/stable/imputation.html

        Assumes ordinal columns values are numerically encoded and imputed.

        Parameters
        ----------
        response: str
            Column name for response variable. Missing values in this column
            will not be imputed

        Returns
        -------
        copy: DataFramePlus
            copy of self with quantitative column values imputed

        """
        # update column kinds
        self.update_col_kinds(self.col_kinds)

        copy = self.data.copy()

        # remove response if it is categorical or quantitative
        quant_cols = (copy.col_kinds['quant'] + copy.col_kinds['ord'])
        try:
            cat_cols.remove(response)
        except ValueError:
            pass
        try:
            quant_cols.remove(response)
        except ValueError:
            pass

        # one-hot encode categorical columns
        df = pd.get_dummies(copy, columns=cat_cols)

        # scale
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)

        # get new df of imputed missing quantitative values
        imp = MICEData(df, perturbation_method='boot')
        imp.update_all(n_iter=n_iter)

        # set missing values in original
        copy.loc[:, quant_cols] = scaler.inverse_transform(imp.data[
                                                           quant_cols])
        # round to avoid precision errors
        copy = copy.round()

        return copy

    def get_model_data(self, response, *args, **kwargs):
        """Prepare full dataframe for modeling.

        Assumes ordinal column values have been encoded and all missing
        values have been imputed.

        Parameters
        ----------
        response: str
            Column name for response variable.

        Returns
        -------
        copy: DataFramePlus
            copy of dataframe with all categorical columns one-hot encoded and
            all columns scaled

        """
        copy = self.data.copy()
        resp = copy[response]
        copy.drop(columns=response)

        # one-hot encode categorical variables
        copy = pd.get_dummies(copy, columns=self.col_kinds['cat'])

        # scale
        sc = MinMaxScaler()
        copy.loc['train'] = sc.fit_transform(copy.loc['train'])
        copy.loc['test'] = sc.transform(copy.loc['test'])
        copy.loc[:, response] = resp
        return copy


class HPDataFramePlus(DataFramePlus):
    """Subclass of DataFramePlus for this dataset.

    Parameters
    ----------
    desc: DataDescription, default None
        Description of dataset

    Attributes
    ----------
    desc: DataDescription
        Description of dataset

    """

    def __init__(self, desc={}, *args, **kwargs):
        """Class constructor."""
        DataFramePlus.__init__(self, *args, **kwargs)
        self._desc = desc

    # override superclass constructor to get class instances from its methods
    @property
    def _constructor(self):
        return HPDataFramePlus

    @property
    def desc(self):
        """Get desc attribute."""
        return self._desc

    @desc.setter
    def desc(self, desc):
        """Set desc attribute."""
        self._desc = desc

    @staticmethod
    def read_csv_with_dtypes(file_path, **kwargs):
        """Read csv into dataframe with dtypes."""
        # Read types first line of csv
        dtypes = pd.read_csv(file_path, nrows=1, **kwargs).iloc[0].to_dict()
        # Read the rest of the lines with the types from above
        return pd.read_csv(file_path, dtype=dtypes, skiprows=[1], **kwargs)

    def print_desc(self, cols=None):
        """Print descriptions of columns.

        Parameters
        ----------
        cols: list, default None
            List of column names. If None full description is printed.

        Raises
        ------
        ValueError:
            desc attribute is empty

        """
        if self.desc:
            # print all columns if none are passed
            if cols is None:
                cols = self.data.columns

            # view description of all variables except sale price
            for col in cols:
                try:
                    print(col + ':' + self.desc[col]['Description'] + '\n')
                except KeyError:
                    pass
                try:
                    for val in self.desc[col]['Values']:
                        print('\t', val + ' - ' + self.desc[col]['Values'][val]
                              )
                # skip over problematic entries
                except KeyError:
                    pass
                except TypeError:
                    pass
                print('\n')
        else:
            raise ValueError('description attribute is empty, set to ' +
                             'DataDescription object')

    def drop_problems(self):
        """Drop problematic columns from this dataset.

        Returns
        -------
        copy: HPDataFramePlus
            copy with problematic columns and observations removed

        """
        # update column kinds
        self.update_col_kinds(self.col_kinds)

        copy = self.data.copy()

        # drop columns with more than 20% of values missing
        notna_col_mask = ~ (copy.isna().sum()/len(copy) > 0.20)
        notna_col_mask.loc['SalePrice'] = True
        copy = copy.loc[:, notna_col_mask]

        # drop outliers in OverallQual
        copy = copy.drop(copy[(copy['OverallQual'] < 5) &
                              (copy['SalePrice'] > 200000)].index)

        # drop outliers in GrLivArea
        copy = copy.drop(copy[(copy['GrLivArea'] > 4000) &
                              (copy['SalePrice'] < 300000)].index)

        # correct erroneous entry
        copy.loc[('test', 2593), 'GarageYrBlt'] = 2007

        return copy

    def hand_impute(self):
        """Impute columns with <= 4 missing values.

        Returns
        -------
        copy: HPDataFramePlus
            copy with problematic columns removed

        """
        # update column kinds
        self.update_col_kinds(self.col_kinds)

        copy = self.data.copy()

        # get names of columns with missing values
        na_cols = self.na_cols_by_kind()

        # get number of missing values for each column kind
        na_counts = self.na_counts()
        cat_na_counts = na_counts[na_cols['cat']]
        ord_na_counts = na_counts[na_cols['ord']]
        quant_na_counts = na_counts[na_cols['quant']]

        # get column kinds with low number of missing values
        cat_lo_na_cols = cat_na_counts[cat_na_counts <= 4].index
        ord_lo_na_cols = ord_na_counts[ord_na_counts <= 2].index
        quant_lo_na_cols = quant_na_counts[quant_na_counts <= 1].index

        # fill low missing value categoricals with mode, quants with median
        missing_values = {**{col: copy[col].mode().values[0] for col
                             in cat_lo_na_cols},
                          **{col: copy[col].median() for col in
                             ord_lo_na_cols.union(quant_lo_na_cols)}}
        copy = copy.fillna(value=missing_values)

        return copy

    def impute_quants(self, response, *args, **kwargs):
        """Impute all missing quantitative column values using MICE.

        See https://www.statsmodels.org/stable/imputation.html

        Overrides superclass method. Assumes ordinal columns values are
        numerically encoded and categorical values are imputed.

        Parameters
        ----------
        response: str
            Column name for response variable. Missing values in this column
            will not be imputed

        Returns
        -------
        copy: HPDataFramePlus
            copy of self with quantitative column values imputed

        """
        # update column kinds
        self.update_col_kinds(self.col_kinds)

        copy = self.data.copy()

        # group column names by type
        cat_cols = self.col_kinds['cat']
        quant_cols = (self.col_kinds['quant'] + self.col_kinds['ord'])

        # drop response
        try:
            cat_cols.remove(response)
        except ValueError:
            pass
        try:
            quant_cols.remove(response)
        except ValueError:
            pass

        # one-hot encode categoricals
        df = pd.get_dummies(copy, columns=cat_cols)

        # rename columns for compatibility with statsmodels formula api
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

        # scale
        scaler = MinMaxScaler()
        df.loc[:, :] = scaler.fit_transform(df)

        # iteratively impute missing values with MICE
        imp = MICEData(df, perturbation_method='boot')
        imp.update_all(n_iter=6)

        # map new column names back to old column names
        inv_rename_cols = {rename_cols[key]: key for key in rename_cols}
        df = df.rename(index=str, columns=inv_rename_cols)
        imp.data = imp.data.rename(index=str, columns=inv_rename_cols)

        # invert scaling for imputed quantitative values
        df.loc[:, :] = scaler.inverse_transform(imp.data)
        copy.loc[:, quant_cols] = df.loc[:, quant_cols].values

        # round to avoid precision errors
        copy = copy.round()

        return copy

    def to_csv_with_dtypes(self, file_path, **kwargs):
        """Save dataframe to csv with dtypes in first row."""
        dtypes_row = pd.DataFrame(self.data.dtypes).T
        dtypes_row.index = pd.MultiIndex.from_tuples([('train', 0)],
                                                     names=['', 'Id'])
        df = pd.concat([dtypes_row, self.data])
        df.to_csv(file_path, **kwargs)


class DataPlus:
    """Augmented dictionary for collections of DataFrames.

    Parameters
    ----------
    dfs: dict, default empty
        Dictionary with keys names (str) and values DataFrames

    Attributes
    ----------
    dfs: dict, default empty
        Dictionary with keys names (str) and values DataFrames

    """

    def __init__(self, dfs={}):
        """Class constructor."""
        self._dfs = dfs

    @property
    def dfs(self):
        """Get dfs attribute."""
        return self._dfs

    @dfs.setter
    def dfs(self, dfs):
        """Set dfs attribute."""
        self._dfs = dfs

    @dfs.deleter
    def dfs(self):
        """Delete dfs attribute."""
        del self._dfs

    def add_dfs(self, new_dfs, dfs_names=None):
        """Add DataFrames to dfs attribute.

        Parameters
        ----------
        new_dfs: DataFrame or iterable of DataFrames
            If DataFrame then df
            If new_dfs is a dictionary, its keys will be used as keys
            in dfs attribute. If other iterable, then df_names will be
            used. If dfs_names is None, names will be generated by default.
        dfs_names: name or iterable of names for DataFrames, default None
            Names for new dfs being added. Pass a single value is new_dfs is
            a DataFrame, or iterable if new_dfs is an iterable. Must consist
            of hashable types.

        Raises
        ------
        TypeError
            df name or names weren't hashable types
        AssertionError


        """
        if isinstance(new_dfs, DataFramePlus):
            if not dfs_names:
                default_name = "df" + str(len(self.dfs))
                self.dfs[default_name] = new_dfs
            try:
                self.dfs[dfs_names] = new_dfs
            except TypeError as e:
                e.args += ('df_names was type' + str(type(dfs_names)))
                raise e.with_traceback(e.__traceback__)
            return

        if isinstance(new_dfs, dict):
            try:
                for key in new_dfs:
                    assert (isinstance(new_dfs[key], DataFramePlus))
                    self.dfs[key] = new_dfs[key]
            except AssertionError as e:
                e.args += ('dfs should all be DataFrames',)
                raise e.with_traceback(e.__traceback__)
            return

        try:
            assert(len(new_dfs) == len(dfs_names))
            new_dfs_iter = iter(new_dfs)
        except TypeError as e:
            e.args += ('new_dfs should be iterable',)
            raise e.with_traceback(e.__traceback__)
        except AssertionError as e:
            e.args += ('new_dfs and dfs_names must be the same length',)
            raise e.with_traceback(e.__traceback__)

        try:
            for (i, df) in enumerate(new_dfs_iter):
                assert(isinstance(df, DataFramePlus))
                self.dfs[dfs_names[i]] = df
        except AssertionError as e:
            e.args += ('dfs should all be DataFrames',)
            raise e.with_traceback(e.__traceback__)
        except TypeError as e:
            e.args += ('df_names must consist of hashable types',)
            raise e.with_traceback(e.__traceback__)
        return

    def save_dfs(self, save_dir):
        """Save dfs data to disk as .csv with default names.

        Parameters
        ----------
        save_dir: str
            directory to save to.

        """
        if self.dfs:
            try:
                for df_name in self.dfs:
                    name = df_name + '.csv'
                    save_path = os.path.join(save_dir, name)
                    self.dfs[df_name].to_csv_with_dtypes(save_path)
            except Exception as e:
                raise e


def combine_MSSubClass(x):
    """Recode values of MSSubClass."""
    if x == 150:
        return 50
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
        return 'Stn_or_Cblk'
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


def combine_cat_vars(data, combine_funcs):
    """Recode categorical variables."""
    copy = data.copy()
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
    copy['HasBsmt'] = (clean_data['BsmtQual'] != 0)
    copy['HasFireplace'] = (clean_data['FireplaceQu'] != 0)
    copy['HasPool'] = (clean_data['PoolQC'] != 0)
    copy['HasGarage'] = (clean_data['GarageQual'] != 0)
    copy['HasFence'] = (clean_data['Fence'] != 0)

    return copy


def create_quant_vars(edit_data, clean_data):
    """Engineer new features from quantitative variables."""
    copy = edit_data.copy()

    # create indicator variables
    copy['Has2ndFlr'] = (clean_data['2ndFlrSF'] != 0)
    copy['HasWoodDeck'] = (clean_data['FireplaceQu'] != 0)
    copy['HasPorch'] = ((clean_data['OpenPorchSF'] != 0) |
                        (clean_data['EnclosedPorch'] != 0) |
                        (clean_data['3SsnPorch'] != 0) |
                        (clean_data['ScreenPorch'] != 0))

    # create overall area variable
    copy['OverallArea'] = (copy['LotArea'] + copy['GrLivArea'] +
                           copy['GarageArea'])

    # create lot variable
    copy['LotRatio'] = copy['LotArea'] / copy['LotFrontage']

    return copy


def log_transform(data, log_cols):
    """Apply log transformation to quantitatives."""
    copy = data.copy()
    for col in log_cols:
        copy['log_' + col] = copy[col].apply(lambda x: 0 if x == 0 else
                                             np.log(x))
    copy.drop(columns=log_cols)
    return copy


if __name__ == '__main__':

    data_dir = 'data'

    # Dataframe for full dataset without modification
    start = time.time()
    print("Creating and processing original dataframe 'orig'")
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col='Id')
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col='Id')
    full = pd.concat([train, test], keys=['train', 'test'], axis=0,
                     sort=False)
    orig = HPDataFramePlus(data=full)

    # sort column names by kind
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
                'PavedDrive', 'PoolQC', 'Fence', 'MoSold', 'YrSold']
    quant_cols = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
                  'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                  'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                  'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF',
                  'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                  'PoolArea', 'MiscVal', 'SalePrice']
    col_kinds = {'cat': cat_cols, 'ord': ord_cols, 'quant': quant_cols}

    # store all column kinds of original df
    orig.col_kinds = col_kinds
    print("Processing of 'orig' complete:", time.time() - start, 'seconds')
    print()

    # Dataframe for cleaned dataset with missing values imputed

    print("Creating and processing cleaned dataframe 'clean'")
    start = time.time()
    # dataset for cleaning
    clean = HPDataFramePlus(data=full)
    # set col kinds of copy
    clean.col_kinds = orig.col_kinds

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
                            orig.data['Functional'].unique()[:-1]))}
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
    ords['Utilities'] = {nan: nan, 'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2,
                         'AllPub': 3}
    # perform encoding
    clean.data = clean.encode_ords(mapper=ords)
    print('\tDropping problematic variables and outliers')
    clean.data = clean.drop_problems()
    # update col kinds
    clean.update_col_kinds(clean.col_kinds)

    print('\tPerforming imputations')
    print('\t\tImputing some missing values by hand')
    clean.data = clean.hand_impute()
    # update col kinds
    clean.update_col_kinds(clean.col_kinds)

    print('\t\tImputing missing categorical values with XGBClassifier')
    clean.data = clean.impute_cats(response='SalePrice')
    print('\t\tImputing missing quantitative values with MICE and PMM')
    clean.data = clean.impute_quants(response='SalePrice')

    print('\t\tSetting dtypes')
    clean.update_col_kinds(col_kinds)
    cats, ords, quants = (clean.col_kinds['cat'], clean.col_kinds['ord'],
                          clean.col_kinds['quant'])
    clean.data.loc[:, cats] = clean.data.loc[:, cats].astype('category')
    clean.data.loc[:, ords] = clean.data.loc[:, ords].astype('int64')
    clean.data.loc[:, 'MSSubClass'] = clean.data['MSSubClass'].astype(
                                      'category')
    clean.data.loc[:, quants] = clean.data.loc[:, quants].astype('float64')
    print()

    print("Processing of 'clean' complete:", time.time() - start, 'seconds')
    print()

    # Save dataframes to disk
    print('\nWriting dfs to .csv files')
    hp_data = DataPlus({'orig': orig, 'clean': clean})
    hp_data.save_dfs(save_dir=data_dir)
    print()

    print('Processing complete. Total runtime {} seconds'.format(time.time
          () - start))
