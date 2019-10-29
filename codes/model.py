


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
        if col != 'SalePrice':
            copy['log_' + col] = copy[col].apply(lambda x: 0 if x == 0 else
                                                 np.log(x))
        else:
            copy['SalePrice'] = np.log(copy['SalePrice'])
    copy.drop(columns=log_cols)
    return copy


if __name__ == '__main__':

    print('\nPerform feature selection and engineering')
    print('\t\nDropping columns\n')
    edit = HPDataFramePlus(data=clean.data)
    drop_cols = ['Heating', 'RoofMatl', 'Condition2', 'Street',
                 'Exterior2nd', 'HouseStyle', 'Utilities', 'PoolQC',
                 '1stFlrSF', 'TotalBsmtSF', 'GarageYrBlt', 'PoolArea',
                 'MiscVal', '3SsnPorch', 'ScreenPorch', 'BsmtFinSF2',
                 'LowQualFinSF']
    edit.data = edit.data.drop(columns=drop_cols)

    print('\t\nCombing categorical feature values\n')
    columns = ['MSSubClass', 'Exterior1st', 'MasVnrType', 'Electrical']
    cat_combine_funcs = {col: None for col in columns}
    cat_combine_funcs['MSSubClass'] = combine_MSSubClass
    cat_combine_funcs['Exterior1st'] = combine_Exterior1st
    cat_combine_funcs['MasVnrType'] = combine_MasVnrType
    cat_combine_funcs['Electrical'] = combine_Electrical
    edit.data = combine_cat_vars(edit.data, cat_combine_funcs)

    print('\t\nCreating new ordinal and quantitative features\n')

    edit.data = create_ord_vars(edit.data, clean.data)
    edit.data = create_quant_vars(edit.data, clean.data)

    print('\t\nLog transforming some quantitative features\n')
    log_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
                'BsmtUnfSF', '2ndFlrSF', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', 'SalePrice', 'OverallArea', 'LotRatio']
    edit.data = log_transform(edit.data, log_cols)

    print('\t\nSaving data\n')
    hp_data = DataPlus({'clean': clean, 'edit': edit})
    data_dir = '../data'
    hp_data.save_dfs(save_dir=data_dir)
    print('\nFeature selection and engineering complete\n')
