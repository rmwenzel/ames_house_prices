---
layout: page
title: Processing and cleaning
---


 The original dataset is available [here](http://www.amstat.org/publications/jse/v19n3/decock/AmesHousing.xls). A version of the dataset is available [on Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). This is the dataset we'll be working with.
 
First we'll do preliminary processing and cleaning of the original dataset. Later we'll [explore the cleaned data and select/engineer features]({{site.baseurl}}/explore/) and [model and predict sale prices]({{site.baseurl}}/model).

<h2>Contents<span class="tocSkip"></span></h2>
<div class="toc"><ul class="toc-item"><li><span><a href="#Setup" data-toc-modified-id="Setup-1">Setup</a></span></li><li><span><a href="#Load-and-inspect-Data" data-toc-modified-id="Load-and-inspect-Data-2">Load and inspect Data</a></span><ul class="toc-item"><li><span><a href="#Variable-descriptions" data-toc-modified-id="Variable-descriptions-2.1">Variable descriptions</a></span></li><li><span><a href="#Load-into-DataFrame" data-toc-modified-id="Load-into-DataFrame-2.2">Load into <code>DataFrame</code></a></span></li></ul></li><li><span><a href="#Clean-data" data-toc-modified-id="Clean-data-3">Clean data</a></span><ul class="toc-item"><li><span><a href="#Classify-variables-by-kind" data-toc-modified-id="Classify-variables-by-kind-3.1">Classify variables by kind</a></span></li><li><span><a href="#Encode-variables" data-toc-modified-id="Encode-variables-3.2">Encode variables</a></span></li><li><span><a href="#Drop-problematic-variables-and-observations" data-toc-modified-id="Drop-problematic-variables-and-observations-3.3">Drop problematic variables and observations</a></span></li><li><span><a href="#Missing-Values" data-toc-modified-id="Missing-Values-3.4">Missing Values</a></span><ul class="toc-item"><li><span><a href="#Inspect-train-and-test-distributions-of-missing-values" data-toc-modified-id="Inspect-train-and-test-distributions-of-missing-values-3.4.1">Inspect train and test distributions of missing values</a></span></li><li><span><a href="#Impute-small-numbers-of-missing-values-by-hand" data-toc-modified-id="Impute-small-numbers-of-missing-values-by-hand-3.4.2">Impute small numbers of missing values by hand</a></span></li><li><span><a href="#Impute-missing-categorical-values-with-XGBClassifier" data-toc-modified-id="Impute-missing-categorical-values-with-XGBClassifier-3.4.3">Impute missing categorical values with <code>XGBClassifier</code></a></span></li><li><span><a href="#Impute-missing-quantitative-values-with-MICE-and-PMM" data-toc-modified-id="Impute-missing-quantitative-values-with-MICE-and-PMM-3.4.4">Impute missing quantitative values with MICE and PMM</a></span></li></ul></li><li><span><a href="#Enforce-dtypes" data-toc-modified-id="Enforce-dtypes-3.5">Enforce dtypes</a></span></li></ul></li><li><span><a href="#Save-processed-data" data-toc-modified-id="Save-processed-data-4">Save processed data</a></span></li></ul></div>

## Setup


```python
# standard imports
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style('whitegrid')
import warnings

import os
import sys

warnings.filterwarnings('ignore')
# add parent directory for importing custom classes
pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(pardir)

# custom class for data description
from codes.process import *
from numpy import nan
```

## Load and inspect Data

### Variable descriptions

A description of the dataset variables is available in `data/data_description.txt`, but it requires a little bit of preprocessing. The custom augmented `dict` class `DataDescription` contains code to do this (see `house_prices/codes/preprocess.py`)


```python
desc = DataDescription('../data/data_description.txt')
```


```python
# First five variable names
list(desc.keys())[:5]
```




    ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street']




```python
# First variable description and values
desc['MSSubClass']
```




    {'Description': ' Identifies the type of dwelling involved in the sale.',
     'Values': {'20': '1-STORY 1946 & NEWER ALL STYLES',
      '30': '1-STORY 1945 & OLDER',
      '40': '1-STORY W/FINISHED ATTIC ALL AGES',
      '45': '1-1/2 STORY - UNFINISHED ALL AGES',
      '50': '1-1/2 STORY FINISHED ALL AGES',
      '60': '2-STORY 1946 & NEWER',
      '70': '2-STORY 1945 & OLDER',
      '75': '2-1/2 STORY ALL AGES',
      '80': 'SPLIT OR MULTI-LEVEL',
      '85': 'SPLIT FOYER',
      '90': 'DUPLEX - ALL STYLES AND AGES',
      '120': '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
      '150': '1-1/2 STORY PUD - ALL AGES',
      '160': '2-STORY PUD - 1946 & NEWER',
      '180': 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
      '190': '2 FAMILY CONVERSION - ALL STYLES AND AGES'}}



### Load into `DataFrame`

We'll combine training and test data into a single `DataFrame`


```python
train = pd.read_csv('../data/train.csv', index_col='Id')
test = pd.read_csv('../data/test.csv', index_col='Id')
full = pd.concat([train, test], keys=['train', 'test'], axis=0, sort=False)
```


```python
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
    <tr>
      <th></th>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">train</td>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 2919 entries, (train, 1) to (test, 2919)
    Data columns (total 80 columns):
    MSSubClass       2919 non-null int64
    MSZoning         2915 non-null object
    LotFrontage      2433 non-null float64
    LotArea          2919 non-null int64
    Street           2919 non-null object
    Alley            198 non-null object
    LotShape         2919 non-null object
    LandContour      2919 non-null object
    Utilities        2917 non-null object
    LotConfig        2919 non-null object
    LandSlope        2919 non-null object
    Neighborhood     2919 non-null object
    Condition1       2919 non-null object
    Condition2       2919 non-null object
    BldgType         2919 non-null object
    HouseStyle       2919 non-null object
    OverallQual      2919 non-null int64
    OverallCond      2919 non-null int64
    YearBuilt        2919 non-null int64
    YearRemodAdd     2919 non-null int64
    RoofStyle        2919 non-null object
    RoofMatl         2919 non-null object
    Exterior1st      2918 non-null object
    Exterior2nd      2918 non-null object
    MasVnrType       2895 non-null object
    MasVnrArea       2896 non-null float64
    ExterQual        2919 non-null object
    ExterCond        2919 non-null object
    Foundation       2919 non-null object
    BsmtQual         2838 non-null object
    BsmtCond         2837 non-null object
    BsmtExposure     2837 non-null object
    BsmtFinType1     2840 non-null object
    BsmtFinSF1       2918 non-null float64
    BsmtFinType2     2839 non-null object
    BsmtFinSF2       2918 non-null float64
    BsmtUnfSF        2918 non-null float64
    TotalBsmtSF      2918 non-null float64
    Heating          2919 non-null object
    HeatingQC        2919 non-null object
    CentralAir       2919 non-null object
    Electrical       2918 non-null object
    1stFlrSF         2919 non-null int64
    2ndFlrSF         2919 non-null int64
    LowQualFinSF     2919 non-null int64
    GrLivArea        2919 non-null int64
    BsmtFullBath     2917 non-null float64
    BsmtHalfBath     2917 non-null float64
    FullBath         2919 non-null int64
    HalfBath         2919 non-null int64
    BedroomAbvGr     2919 non-null int64
    KitchenAbvGr     2919 non-null int64
    KitchenQual      2918 non-null object
    TotRmsAbvGrd     2919 non-null int64
    Functional       2917 non-null object
    Fireplaces       2919 non-null int64
    FireplaceQu      1499 non-null object
    GarageType       2762 non-null object
    GarageYrBlt      2760 non-null float64
    GarageFinish     2760 non-null object
    GarageCars       2918 non-null float64
    GarageArea       2918 non-null float64
    GarageQual       2760 non-null object
    GarageCond       2760 non-null object
    PavedDrive       2919 non-null object
    WoodDeckSF       2919 non-null int64
    OpenPorchSF      2919 non-null int64
    EnclosedPorch    2919 non-null int64
    3SsnPorch        2919 non-null int64
    ScreenPorch      2919 non-null int64
    PoolArea         2919 non-null int64
    PoolQC           10 non-null object
    Fence            571 non-null object
    MiscFeature      105 non-null object
    MiscVal          2919 non-null int64
    MoSold           2919 non-null int64
    YrSold           2919 non-null int64
    SaleType         2918 non-null object
    SaleCondition    2919 non-null object
    SalePrice        1460 non-null float64
    dtypes: float64(12), int64(25), object(43)
    memory usage: 1.8+ MB


We can see some cleanup and preprocessing will be necessary. For example, there are quite a few missing values, and more than half the variables have been have been cast to `pandas` catch-all `object` dtype.

About half of the data is training data and half is testing data - observations from the testing data have `NaN` values for `SalePrice`


```python
# shape of training data
full.loc['train'].shape
```




    (1460, 80)




```python
# shape of training data
full.loc['test'].shape
```




    (1459, 80)



## Clean data

Note: all the functions in [this section](#Drop-problematic-variables-and-observations) are rolled into `HPDataFramePlus` methods `encode_ords, drop_probs`

Before we clean any data, we'll store the original dataset so we have an unadulaterated copy


```python
# create instance of HPDataFramePlus for full dataset
orig = HPDataFramePlus(data=full)
```

### Classify variables by kind

Since it doesn't affect the data, we'll group variables in the original dataset into categorical, ordinal and quantitative kinds. We'll use the custom class `HPDataFramePlus` which contains helpful methods


```python
# set description attribute
orig.desc = desc

# view description of all variables except sale price
cols = list(full.columns)
cols.remove('SalePrice')
orig.print_desc(cols)
```

    MSSubClass: Identifies the type of dwelling involved in the sale.
    
    	 20 - 1-STORY 1946 & NEWER ALL STYLES
    	 30 - 1-STORY 1945 & OLDER
    	 40 - 1-STORY W/FINISHED ATTIC ALL AGES
    	 45 - 1-1/2 STORY - UNFINISHED ALL AGES
    	 50 - 1-1/2 STORY FINISHED ALL AGES
    	 60 - 2-STORY 1946 & NEWER
    	 70 - 2-STORY 1945 & OLDER
    	 75 - 2-1/2 STORY ALL AGES
    	 80 - SPLIT OR MULTI-LEVEL
    	 85 - SPLIT FOYER
    	 90 - DUPLEX - ALL STYLES AND AGES
    	 120 - 1-STORY PUD (Planned Unit Development) - 1946 & NEWER
    	 150 - 1-1/2 STORY PUD - ALL AGES
    	 160 - 2-STORY PUD - 1946 & NEWER
    	 180 - PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
    	 190 - 2 FAMILY CONVERSION - ALL STYLES AND AGES
    
    
    MSZoning: Identifies the general zoning classification of the sale.
    
    	 A - Agriculture
    	 C - Commercial
    	 FV - Floating Village Residential
    	 I - Industrial
    	 RH - Residential High Density
    	 RL - Residential Low Density
    	 RP - Residential Low Density Park 
    	 RM - Residential Medium Density
    
    
    LotFrontage: Linear feet of street connected to property
    
    
    
    LotArea: Lot size in square feet
    
    
    
    Street: Type of road access to property
    
    	 Grvl - Gravel
    	 Pave - Paved
    
    
    Alley: Type of alley access to property
    
    	 Grvl - Gravel
    	 Pave - Paved
    	 NA - No alley access
    
    
    LotShape: General shape of property
    
    	 Reg - Regular
    	 IR1 - Slightly irregular
    	 IR2 - Moderately Irregular
    	 IR3 - Irregular
    
    
    LandContour: Flatness of the property
    
    	 Lvl - Near Flat/Level
    	 Bnk - Banked - Quick and significant rise from street grade to building
    	 HLS - Hillside - Significant slope from side to side
    	 Low - Depression
    
    
    Utilities: Type of utilities available
    
    	 AllPub - All public Utilities (E,G,W,& S)
    	 NoSewr - Electricity, Gas, and Water (Septic Tank)
    	 NoSeWa - Electricity and Gas Only
    	 ELO - Electricity only
    
    
    LotConfig: Lot configuration
    
    	 Inside - Inside lot
    	 Corner - Corner lot
    	 CulDSac - Cul-de-sac
    	 FR2 - Frontage on 2 sides of property
    	 FR3 - Frontage on 3 sides of property
    
    
    LandSlope: Slope of property
    
    	 Gtl - Gentle slope
    	 Mod - Moderate Slope
    	 Sev - Severe Slope
    
    
    Neighborhood: Physical locations within Ames city limits
    
    	 Blmngtn - Bloomington Heights
    	 Blueste - Bluestem
    	 BrDale - Briardale
    	 BrkSide - Brookside
    	 ClearCr - Clear Creek
    	 CollgCr - College Creek
    	 Crawfor - Crawford
    	 Edwards - Edwards
    	 Gilbert - Gilbert
    	 IDOTRR - Iowa DOT and Rail Road
    	 MeadowV - Meadow Village
    	 Mitchel - Mitchell
    	 Names - North Ames
    	 NoRidge - Northridge
    	 NPkVill - Northpark Villa
    	 NridgHt - Northridge Heights
    	 NWAmes - Northwest Ames
    	 OldTown - Old Town
    	 SWISU - South & West of Iowa State University
    	 Sawyer - Sawyer
    	 SawyerW - Sawyer West
    	 Somerst - Somerset
    	 StoneBr - Stone Brook
    	 Timber - Timberland
    	 Veenker - Veenker
    
    
    Condition1: Proximity to various conditions
    
    	 Artery - Adjacent to arterial street
    	 Feedr - Adjacent to feeder street
    	 Norm - Normal
    	 RRNn - Within 200' of North-South Railroad
    	 RRAn - Adjacent to North-South Railroad
    	 PosN - Near positive off-site feature--park, greenbelt, etc.
    	 PosA - Adjacent to postive off-site feature
    	 RRNe - Within 200' of East-West Railroad
    	 RRAe - Adjacent to East-West Railroad
    
    
    Condition2: Proximity to various conditions (if more than one is present)
    
    	 Artery - Adjacent to arterial street
    	 Feedr - Adjacent to feeder street
    	 Norm - Normal
    	 RRNn - Within 200' of North-South Railroad
    	 RRAn - Adjacent to North-South Railroad
    	 PosN - Near positive off-site feature--park, greenbelt, etc.
    	 PosA - Adjacent to postive off-site feature
    	 RRNe - Within 200' of East-West Railroad
    	 RRAe - Adjacent to East-West Railroad
    
    
    BldgType: Type of dwelling
    
    	 1Fam - Single-family Detached
    	 2FmCon - Two-family Conversion; originally built as one-family dwelling
    	 Duplx - Duplex
    	 TwnhsE - Townhouse End Unit
    	 TwnhsI - Townhouse Inside Unit
    
    
    HouseStyle: Style of dwelling
    
    	 1Story - One story
    	 1.5Fin - One and one-half story: 2nd level finished
    	 1.5Unf - One and one-half story: 2nd level unfinished
    	 2Story - Two story
    	 2.5Fin - Two and one-half story: 2nd level finished
    	 2.5Unf - Two and one-half story: 2nd level unfinished
    	 SFoyer - Split Foyer
    	 SLvl - Split Level
    
    
    OverallQual: Rates the overall material and finish of the house
    
    	 10 - Very Excellent
    	 9 - Excellent
    	 8 - Very Good
    	 7 - Good
    	 6 - Above Average
    	 5 - Average
    	 4 - Below Average
    	 3 - Fair
    	 2 - Poor
    	 1 - Very Poor
    
    
    OverallCond: Rates the overall condition of the house
    
    	 10 - Very Excellent
    	 9 - Excellent
    	 8 - Very Good
    	 7 - Good
    	 6 - Above Average
    	 5 - Average
    	 4 - Below Average
    	 3 - Fair
    	 2 - Poor
    	 1 - Very Poor
    
    
    YearBuilt: Original construction date
    
    
    
    YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
    
    
    
    RoofStyle: Type of roof
    
    	 Flat - Flat
    	 Gable - Gable
    	 Gambrel - Gabrel (Barn)
    	 Hip - Hip
    	 Mansard - Mansard
    	 Shed - Shed
    
    
    RoofMatl: Roof material
    
    	 ClyTile - Clay or Tile
    	 CompShg - Standard (Composite) Shingle
    	 Membran - Membrane
    	 Metal - Metal
    	 Roll - Roll
    	 Tar&Grv - Gravel & Tar
    	 WdShake - Wood Shakes
    	 WdShngl - Wood Shingles
    
    
    Exterior1st: Exterior covering on house
    
    	 AsbShng - Asbestos Shingles
    	 AsphShn - Asphalt Shingles
    	 BrkComm - Brick Common
    	 BrkFace - Brick Face
    	 CBlock - Cinder Block
    	 CemntBd - Cement Board
    	 HdBoard - Hard Board
    	 ImStucc - Imitation Stucco
    	 MetalSd - Metal Siding
    	 Other - Other
    	 Plywood - Plywood
    	 PreCast - PreCast
    	 Stone - Stone
    	 Stucco - Stucco
    	 VinylSd - Vinyl Siding
    	 Sdng - Wood Siding
    	 WdShing - Wood Shingles
    
    
    Exterior2nd: Exterior covering on house (if more than one material)
    
    	 AsbShng - Asbestos Shingles
    	 AsphShn - Asphalt Shingles
    	 BrkComm - Brick Common
    	 BrkFace - Brick Face
    	 CBlock - Cinder Block
    	 CemntBd - Cement Board
    	 HdBoard - Hard Board
    	 ImStucc - Imitation Stucco
    	 MetalSd - Metal Siding
    	 Other - Other
    	 Plywood - Plywood
    	 PreCast - PreCast
    	 Stone - Stone
    	 Stucco - Stucco
    	 VinylSd - Vinyl Siding
    	 Sdng - Wood Siding
    	 WdShing - Wood Shingles
    
    
    MasVnrType: Masonry veneer type
    
    	 BrkCmn - Brick Common
    	 BrkFace - Brick Face
    	 CBlock - Cinder Block
    	 None - None
    	 Stone - Stone
    
    
    MasVnrArea: Masonry veneer area in square feet
    
    
    
    ExterQual: Evaluates the quality of the material on the exterior 
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Average/Typical
    	 Fa - Fair
    	 Po - Poor
    
    
    ExterCond: Evaluates the present condition of the material on the exterior
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Average/Typical
    	 Fa - Fair
    	 Po - Poor
    
    
    Foundation: Type of foundation
    
    	 BrkTil - Brick & Tile
    	 CBlock - Cinder Block
    	 PConc - Poured Contrete
    	 Slab - Slab
    	 Stone - Stone
    	 Wood - Wood
    
    
    BsmtQual: Evaluates the height of the basement
    
    	 Ex - Excellent (100+ inches)
    	 Gd - Good (90-99 inches)
    	 TA - Typical (80-89 inches)
    	 Fa - Fair (70-79 inches)
    	 Po - Poor (<70 inches
    	 NA - No Basement
    
    
    BsmtCond: Evaluates the general condition of the basement
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Typical - slight dampness allowed
    	 Fa - Fair - dampness or some cracking or settling
    	 Po - Poor - Severe cracking, settling, or wetness
    	 NA - No Basement
    
    
    BsmtExposure: Refers to walkout or garden level walls
    
    	 Gd - Good Exposure
    	 Av - Average Exposure (split levels or foyers typically score average or above)
    	 Mn - Mimimum Exposure
    	 No - No Exposure
    	 NA - No Basement
    
    
    BsmtFinType1: Rating of basement finished area
    
    	 GLQ - Good Living Quarters
    	 ALQ - Average Living Quarters
    	 BLQ - Below Average Living Quarters
    	 Rec - Average Rec Room
    	 LwQ - Low Quality
    	 Unf - Unfinshed
    	 NA - No Basement
    
    
    BsmtFinSF1: Type 1 finished square feet
    
    
    
    BsmtFinType2: Rating of basement finished area (if multiple types)
    
    	 GLQ - Good Living Quarters
    	 ALQ - Average Living Quarters
    	 BLQ - Below Average Living Quarters
    	 Rec - Average Rec Room
    	 LwQ - Low Quality
    	 Unf - Unfinshed
    	 NA - No Basement
    
    
    BsmtFinSF2: Type 2 finished square feet
    
    
    
    BsmtUnfSF: Unfinished square feet of basement area
    
    
    
    TotalBsmtSF: Total square feet of basement area
    
    
    
    Heating: Type of heating
    
    	 Floor - Floor Furnace
    	 GasA - Gas forced warm air furnace
    	 GasW - Gas hot water or steam heat
    	 Grav - Gravity furnace
    	 OthW - Hot water or steam heat other than gas
    	 Wall - Wall furnace
    
    
    HeatingQC: Heating quality and condition
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Average/Typical
    	 Fa - Fair
    	 Po - Poor
    
    
    CentralAir: Central air conditioning
    
    	 N - No
    	 Y - Yes
    
    
    Electrical: Electrical system
    
    	 SBrkr - Standard Circuit Breakers & Romex
    	 FuseA - Fuse Box over 60 AMP and all Romex wiring (Average)
    	 FuseF - 60 AMP Fuse Box and mostly Romex wiring (Fair)
    	 FuseP - 60 AMP Fuse Box and mostly knob & tube wiring (poor)
    	 Mix - Mixed
    
    
    1stFlrSF: First Floor square feet
    
    
    
    2ndFlrSF: Second floor square feet
    
    
    
    LowQualFinSF: Low quality finished square feet (all floors)
    
    
    
    GrLivArea: Above grade (ground) living area square feet
    
    
    
    BsmtFullBath: Basement full bathrooms
    
    
    
    BsmtHalfBath: Basement half bathrooms
    
    
    
    FullBath: Full bathrooms above grade
    
    
    
    HalfBath: Half baths above grade
    
    
    
    BedroomAbvGr: Bedrooms above grade (does NOT include basement bedrooms)
    
    
    
    KitchenAbvGr: Kitchens above grade
    
    
    
    KitchenQual: Kitchen quality
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Typical/Average
    	 Fa - Fair
    	 Po - Poor
    
    
    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    
    
    
    Functional: Home functionality (Assume typical unless deductions are warranted)
    
    	 Typ - Typical Functionality
    	 Min1 - Minor Deductions 1
    	 Min2 - Minor Deductions 2
    	 Mod - Moderate Deductions
    	 Maj1 - Major Deductions 1
    	 Maj2 - Major Deductions 2
    	 Sev - Severely Damaged
    	 Sal - Salvage only
    
    
    Fireplaces: Number of fireplaces
    
    
    
    FireplaceQu: Fireplace quality
    
    	 Ex - Excellent - Exceptional Masonry Fireplace
    	 Gd - Good - Masonry Fireplace in main level
    	 TA - Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
    	 Fa - Fair - Prefabricated Fireplace in basement
    	 Po - Poor - Ben Franklin Stove
    	 NA - No Fireplace
    
    
    GarageType: Garage location
    
    	 2Types - More than one type of garage
    	 Attchd - Attached to home
    	 Basment - Basement Garage
    	 BuiltIn - Built-In (Garage part of house - typically has room above garage)
    	 CarPort - Car Port
    	 Detchd - Detached from home
    	 NA - No Garage
    
    
    GarageYrBlt: Year garage was built
    
    
    
    GarageFinish: Interior finish of the garage
    
    	 Fin - Finished
    	 RFn - Rough Finished
    	 Unf - Unfinished
    	 NA - No Garage
    
    
    GarageCars: Size of garage in car capacity
    
    
    
    GarageArea: Size of garage in square feet
    
    
    
    GarageQual: Garage quality
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Typical/Average
    	 Fa - Fair
    	 Po - Poor
    	 NA - No Garage
    
    
    GarageCond: Garage condition
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Typical/Average
    	 Fa - Fair
    	 Po - Poor
    	 NA - No Garage
    
    
    PavedDrive: Paved driveway
    
    	 Y - Paved 
    	 P - Partial Pavement
    	 N - Dirt/Gravel
    
    
    WoodDeckSF: Wood deck area in square feet
    
    
    
    OpenPorchSF: Open porch area in square feet
    
    
    
    EnclosedPorch: Enclosed porch area in square feet
    
    
    
    3SsnPorch: Three season porch area in square feet
    
    
    
    ScreenPorch: Screen porch area in square feet
    
    
    
    PoolArea: Pool area in square feet
    
    
    
    PoolQC: Pool quality
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Average/Typical
    	 Fa - Fair
    	 NA - No Pool
    
    
    Fence: Fence quality
    
    	 GdPrv - Good Privacy
    	 MnPrv - Minimum Privacy
    	 GdWo - Good Wood
    	 MnWw - Minimum Wood/Wire
    	 NA - No Fence
    
    
    MiscFeature: Miscellaneous feature not covered in other categories
    
    	 Elev - Elevator
    	 Gar2 - 2nd Garage (if not described in garage section)
    	 Othr - Other
    	 Shed - Shed (over 100 SF)
    	 TenC - Tennis Court
    	 NA - None
    
    
    MiscVal: $Value of miscellaneous feature
    
    
    
    MoSold: Month Sold (MM)
    
    
    
    YrSold: Year Sold (YYYY)
    
    
    
    SaleType: Type of sale
    
    	 WD - Warranty Deed - Conventional
    	 CWD - Warranty Deed - Cash
    	 VWD - Warranty Deed - VA Loan
    	 New - Home just constructed and sold
    	 COD - Court Officer Deed/Estate
    	 Con - Contract 15% Down payment regular terms
    	 ConLw - Contract Low Down payment and low interest
    	 ConLI - Contract Low Interest
    	 ConLD - Contract Low Down
    	 Oth - Other
    
    
    SaleCondition: Condition of sale
    
    	 Normal - Normal Sale
    	 Abnorml - Abnormal Sale -  trade, foreclosure, short sale
    	 AdjLand - Adjoining Land Purchase
    	 Alloca - Allocation - two linked properties with separate deeds, typically condo with a garage unit
    	 Family - Sale between family members
    	 Partial - Home was not completed when last assessed (associated with New Homes)
    
    


To classify the variables, there's really no alternative here than to carefully inspect the variable descriptions and determine which is which. To clarify our terms:

- Categorical variables are discrete variables with no ordering (although they may have a numerical encoding)
- Ordinal variables are discrete numeric variables, hence they have an ordering (and should be numerically encoded)
- Quantiative variables are continuous numeric variables


```python
# split variables into categorical, ordinal, quantitative
cat_cols = ['MSSubClass', 'MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
            'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 
            'Electrical', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition', 'Alley']
ord_cols = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 
            'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'HeatingQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
            'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
            'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
            'MoSold', 'YrSold']
quant_cols = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 
              'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
              'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
              'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
              'SalePrice']

# group columns by kind
col_kinds = {'cat': cat_cols, 'ord': ord_cols, 'quant': quant_cols}

# set col_kinds attribute
orig.col_kinds = col_kinds
```

Now the cleaning begins.


```python
# create new dataframe for cleaned data and set attributes
clean = HPDataFramePlus(data=full)
clean.col_kinds = orig.col_kinds
clean.desc = orig.desc
```

### Encode variables

Before we can clean, we need to make sure all variables are encoded appropriately. Let's compare the dtypes of our dataframe with the variable types


```python
# dtypes for categorical variables
clean.data[clean.col_kinds['cat']].info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 2919 entries, (train, 1) to (test, 2919)
    Data columns (total 24 columns):
    MSSubClass       2919 non-null int64
    MSZoning         2915 non-null object
    Street           2919 non-null object
    LandContour      2919 non-null object
    LotConfig        2919 non-null object
    Neighborhood     2919 non-null object
    Condition1       2919 non-null object
    Condition2       2919 non-null object
    BldgType         2919 non-null object
    HouseStyle       2919 non-null object
    RoofStyle        2919 non-null object
    RoofMatl         2919 non-null object
    Exterior1st      2918 non-null object
    Exterior2nd      2918 non-null object
    MasVnrType       2895 non-null object
    Foundation       2919 non-null object
    Heating          2919 non-null object
    CentralAir       2919 non-null object
    Electrical       2918 non-null object
    GarageType       2762 non-null object
    MiscFeature      105 non-null object
    SaleType         2918 non-null object
    SaleCondition    2919 non-null object
    Alley            198 non-null object
    dtypes: int64(1), object(23)
    memory usage: 578.8+ KB



```python
# dtypes for ordinal variables
clean.data[clean.col_kinds['ord']].info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 2919 entries, (train, 1) to (test, 2919)
    Data columns (total 33 columns):
    LotShape        2919 non-null object
    Utilities       2917 non-null object
    LandSlope       2919 non-null object
    OverallQual     2919 non-null int64
    OverallCond     2919 non-null int64
    ExterQual       2919 non-null object
    ExterCond       2919 non-null object
    BsmtQual        2838 non-null object
    BsmtCond        2837 non-null object
    BsmtExposure    2837 non-null object
    BsmtFinType1    2840 non-null object
    BsmtFinType2    2839 non-null object
    HeatingQC       2919 non-null object
    BsmtFullBath    2917 non-null float64
    BsmtHalfBath    2917 non-null float64
    FullBath        2919 non-null int64
    HalfBath        2919 non-null int64
    BedroomAbvGr    2919 non-null int64
    KitchenAbvGr    2919 non-null int64
    KitchenQual     2918 non-null object
    TotRmsAbvGrd    2919 non-null int64
    Functional      2917 non-null object
    Fireplaces      2919 non-null int64
    FireplaceQu     1499 non-null object
    GarageFinish    2760 non-null object
    GarageCars      2918 non-null float64
    GarageQual      2760 non-null object
    GarageCond      2760 non-null object
    PavedDrive      2919 non-null object
    PoolQC          10 non-null object
    Fence           571 non-null object
    MoSold          2919 non-null int64
    YrSold          2919 non-null int64
    dtypes: float64(3), int64(10), object(20)
    memory usage: 784.1+ KB



```python
# dtypes for quantitative variables
clean.data[clean.col_kinds['quant']].info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 2919 entries, (train, 1) to (test, 2919)
    Data columns (total 23 columns):
    LotFrontage      2433 non-null float64
    LotArea          2919 non-null int64
    YearBuilt        2919 non-null int64
    YearRemodAdd     2919 non-null int64
    MasVnrArea       2896 non-null float64
    BsmtFinSF1       2918 non-null float64
    BsmtFinSF2       2918 non-null float64
    BsmtUnfSF        2918 non-null float64
    TotalBsmtSF      2918 non-null float64
    1stFlrSF         2919 non-null int64
    2ndFlrSF         2919 non-null int64
    LowQualFinSF     2919 non-null int64
    GrLivArea        2919 non-null int64
    GarageYrBlt      2760 non-null float64
    GarageArea       2918 non-null float64
    WoodDeckSF       2919 non-null int64
    OpenPorchSF      2919 non-null int64
    EnclosedPorch    2919 non-null int64
    3SsnPorch        2919 non-null int64
    ScreenPorch      2919 non-null int64
    PoolArea         2919 non-null int64
    MiscVal          2919 non-null int64
    SalePrice        1460 non-null float64
    dtypes: float64(9), int64(14)
    memory usage: 556.0+ KB


Categorical and quantitative dtypes look good, but we'll need to deal with the ordinal variables.


```python
# inspect description of ordinal variables
clean.print_desc(clean.col_kinds['ord'])
```

    LotShape: General shape of property
    
    	 Reg - Regular
    	 IR1 - Slightly irregular
    	 IR2 - Moderately Irregular
    	 IR3 - Irregular
    
    
    Utilities: Type of utilities available
    
    	 AllPub - All public Utilities (E,G,W,& S)
    	 NoSewr - Electricity, Gas, and Water (Septic Tank)
    	 NoSeWa - Electricity and Gas Only
    	 ELO - Electricity only
    
    
    LandSlope: Slope of property
    
    	 Gtl - Gentle slope
    	 Mod - Moderate Slope
    	 Sev - Severe Slope
    
    
    OverallQual: Rates the overall material and finish of the house
    
    	 10 - Very Excellent
    	 9 - Excellent
    	 8 - Very Good
    	 7 - Good
    	 6 - Above Average
    	 5 - Average
    	 4 - Below Average
    	 3 - Fair
    	 2 - Poor
    	 1 - Very Poor
    
    
    OverallCond: Rates the overall condition of the house
    
    	 10 - Very Excellent
    	 9 - Excellent
    	 8 - Very Good
    	 7 - Good
    	 6 - Above Average
    	 5 - Average
    	 4 - Below Average
    	 3 - Fair
    	 2 - Poor
    	 1 - Very Poor
    
    
    ExterQual: Evaluates the quality of the material on the exterior 
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Average/Typical
    	 Fa - Fair
    	 Po - Poor
    
    
    ExterCond: Evaluates the present condition of the material on the exterior
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Average/Typical
    	 Fa - Fair
    	 Po - Poor
    
    
    BsmtQual: Evaluates the height of the basement
    
    	 Ex - Excellent (100+ inches)
    	 Gd - Good (90-99 inches)
    	 TA - Typical (80-89 inches)
    	 Fa - Fair (70-79 inches)
    	 Po - Poor (<70 inches
    	 NA - No Basement
    
    
    BsmtCond: Evaluates the general condition of the basement
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Typical - slight dampness allowed
    	 Fa - Fair - dampness or some cracking or settling
    	 Po - Poor - Severe cracking, settling, or wetness
    	 NA - No Basement
    
    
    BsmtExposure: Refers to walkout or garden level walls
    
    	 Gd - Good Exposure
    	 Av - Average Exposure (split levels or foyers typically score average or above)
    	 Mn - Mimimum Exposure
    	 No - No Exposure
    	 NA - No Basement
    
    
    BsmtFinType1: Rating of basement finished area
    
    	 GLQ - Good Living Quarters
    	 ALQ - Average Living Quarters
    	 BLQ - Below Average Living Quarters
    	 Rec - Average Rec Room
    	 LwQ - Low Quality
    	 Unf - Unfinshed
    	 NA - No Basement
    
    
    BsmtFinType2: Rating of basement finished area (if multiple types)
    
    	 GLQ - Good Living Quarters
    	 ALQ - Average Living Quarters
    	 BLQ - Below Average Living Quarters
    	 Rec - Average Rec Room
    	 LwQ - Low Quality
    	 Unf - Unfinshed
    	 NA - No Basement
    
    
    HeatingQC: Heating quality and condition
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Average/Typical
    	 Fa - Fair
    	 Po - Poor
    
    
    BsmtFullBath: Basement full bathrooms
    
    
    
    BsmtHalfBath: Basement half bathrooms
    
    
    
    FullBath: Full bathrooms above grade
    
    
    
    HalfBath: Half baths above grade
    
    
    
    BedroomAbvGr: Bedrooms above grade (does NOT include basement bedrooms)
    
    
    
    KitchenAbvGr: Kitchens above grade
    
    
    
    KitchenQual: Kitchen quality
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Typical/Average
    	 Fa - Fair
    	 Po - Poor
    
    
    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    
    
    
    Functional: Home functionality (Assume typical unless deductions are warranted)
    
    	 Typ - Typical Functionality
    	 Min1 - Minor Deductions 1
    	 Min2 - Minor Deductions 2
    	 Mod - Moderate Deductions
    	 Maj1 - Major Deductions 1
    	 Maj2 - Major Deductions 2
    	 Sev - Severely Damaged
    	 Sal - Salvage only
    
    
    Fireplaces: Number of fireplaces
    
    
    
    FireplaceQu: Fireplace quality
    
    	 Ex - Excellent - Exceptional Masonry Fireplace
    	 Gd - Good - Masonry Fireplace in main level
    	 TA - Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
    	 Fa - Fair - Prefabricated Fireplace in basement
    	 Po - Poor - Ben Franklin Stove
    	 NA - No Fireplace
    
    
    GarageFinish: Interior finish of the garage
    
    	 Fin - Finished
    	 RFn - Rough Finished
    	 Unf - Unfinished
    	 NA - No Garage
    
    
    GarageCars: Size of garage in car capacity
    
    
    
    GarageQual: Garage quality
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Typical/Average
    	 Fa - Fair
    	 Po - Poor
    	 NA - No Garage
    
    
    GarageCond: Garage condition
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Typical/Average
    	 Fa - Fair
    	 Po - Poor
    	 NA - No Garage
    
    
    PavedDrive: Paved driveway
    
    	 Y - Paved 
    	 P - Partial Pavement
    	 N - Dirt/Gravel
    
    
    PoolQC: Pool quality
    
    	 Ex - Excellent
    	 Gd - Good
    	 TA - Average/Typical
    	 Fa - Fair
    	 NA - No Pool
    
    
    Fence: Fence quality
    
    	 GdPrv - Good Privacy
    	 MnPrv - Minimum Privacy
    	 GdWo - Good Wood
    	 MnWw - Minimum Wood/Wire
    	 NA - No Fence
    
    
    MoSold: Month Sold (MM)
    
    
    
    YrSold: Year Sold (YYYY)
    
    
    


Upon careful reflection, we'll encode the ordinal variable values by hand (taking care to distinguish between values of 0 and truly missing values)


```python
# encode ordinal variable values in dictionary by hand when needed
ords = {}
ords['GarageCond'] = {np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
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
ords['Utilities'] = {nan: nan, 'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}

# perform encoding
clean.data = clean.encode_ords(mapper=ords)
```

### Drop problematic variables and observations

Now that all variables are properly encoded, we can drop those with too many missing values immediately. We'll be somewhat conservative and drop variables missing $>20\%$ of values


```python
def drop_mostly_missing_cols(hpdf):
    """Drop columns with too many missing values"""
    copy = hpdf.data.copy()
    # drop columns with more than 20% values missing
    notna_col_mask = ~ (copy.isna().sum()/len(copy) > 0.20)
    notna_col_mask.loc['SalePrice'] = True
    copy = copy.loc[: , notna_col_mask]
    # drop columns associated with those
    copy.drop(columns=['MiscVal'])
    return copy

# create a new dataframe for cleaning
clean.data = drop_mostly_missing_cols(clean)
clean.update_col_kinds(clean.col_kinds)
```

We'll also drop some well-known outlying observations (at least, well-known on Kaggle ) in the training data. Dropping outliers is (for good reason) very controversial, and one should take great care in doing so. The justification for it depends on context, however. In our case, the end goal is to predict `SalePrice` accuractely. If dropping outliers improves the ability of our prediction model to generalize, than this may provide some retroactive justification.

First we'll plot the outliers (identifying them by their relationship to `SalePrice`)


```python
def plot_outliers(hpdf):
    """Plot variables which contain well-known outliers."""
    plt.subplots(1, 2, figsize=(15, 10))
    train = hpdf.data.loc['train', :]
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='OverallQual', y='SalePrice', data=train)
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)


plot_outliers(clean)
```

![png]({{site.baseurl}}/assets/images/process_42_0.png)


Kagglers seem to frequently conclude that the outliers are the house with overall quality 4 but a sale price of more than \\$250,000, and the two houses with more than 4500 sq ft of general living area but sale prices less than \\$300,000.

Whether this is well-justified, and how much it's an example of groupthink, is a matter for debate. But it seems to regularly improve the predictive capability of models, so we'll follow suit.


```python
def drop_outliers(hpdf):
    copy = hpdf.data.copy()
    # drop outliers in OverallQual
    idx = copy[(copy['OverallQual'] < 5) & (copy['SalePrice'] > 200000)].index[0][1]
    copy = copy.drop(labels=[idx], axis=0, level='Id')
    # drop outliers in GrLivArea
    idx = copy[(copy['GrLivArea'] > 4000) & (copy['SalePrice'] < 300000)].index[0][1]
    copy = copy.drop(labels=[idx], axis=0, level='Id')
    
    return copy

clean.data = drop_outliers(clean)
clean.update_col_kinds(clean.col_kinds)
```

Finally, we'll see if there are any categorical variables with extremely unbalanced distributions


```python
def print_unbal_dists(data, bal_threshold):
    """Print distributions of columns with more than bal_threshold proportion concentrated at a single value."""
    dists = []
    for col in data.columns:
        val_counts = data[col].value_counts()
        dist = val_counts/sum(val_counts)
        if dist.max() > bal_threshold:
            dists += [dist]
    for dist in dists:
        print()
        print(dist)
```

### Missing Values

Our dataset is missing a lot of values!


```python
# counts of missing values by variable, excluding SalePrice
clean.na_counts().drop('SalePrice')
```




    MSZoning          4
    LotFrontage     485
    Utilities         2
    Exterior1st       1
    Exterior2nd       1
    MasVnrType       24
    MasVnrArea       23
    BsmtFinSF1        1
    BsmtFinSF2        1
    BsmtUnfSF         1
    TotalBsmtSF       1
    Electrical        1
    BsmtFullBath      2
    BsmtHalfBath      2
    Functional        2
    GarageType      157
    GarageYrBlt     159
    GarageCars        1
    GarageArea        1
    SaleType          1
    dtype: int64



#### Inspect train and test distributions of missing values

 Before we get into imputing them, to inform our choice of methods, let's see how their distributions might differ across training and test sets. We want to be careful imputing missing values when those missing values are distributed unevenly across train and test sets if our goal is prediction, since our imputation could introduce further difference between train and test sets.


```python
def plot_train_and_test_missing_values(hpdf):
    """plot distribution of missing train values."""
    
    copy = hpdf.data.drop(columns=['SalePrice'])
    train = HPDataFramePlus(data=copy.loc['train', :])
    test = HPDataFramePlus(data=copy.loc['test', :])
    
    fig, _ = plt.subplots(1, 2, figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    train_missing_dist = train.na_counts()/train.na_counts().sum()
    sns.barplot(x=train_missing_dist.index, y=train_missing_dist.values)
    plt.xticks(rotation=75)
    
    plt.subplot(1, 2, 2)
    test_missing_dist = test.na_counts()/test.na_counts().sum()
    sns.barplot(x=test_missing_dist.index, y=test_missing_dist.values)
    plt.xticks(rotation=75)

plot_train_and_test_missing_values(clean)
```

![png]({{site.baseurl}}/assets/images/process_52_0.png)


```python
# variables missing values in train but not test set
train_missing = HPDataFramePlus(data=clean.data.loc['train', :]).na_counts()
test_missing = HPDataFramePlus(data=clean.data.loc['test', :]).na_counts()
train_not_test = list(set(train_missing.index).difference(test_missing.index))
train_not_test
```




    ['Electrical']




```python
# variables missing values in test but not train set
test_not_train = list(set(test_missing.index).difference(train_missing.index))
test_not_train
```




    ['GarageArea',
     'Exterior2nd',
     'BsmtHalfBath',
     'BsmtUnfSF',
     'BsmtFinSF2',
     'BsmtFinSF1',
     'Functional',
     'Utilities',
     'GarageCars',
     'TotalBsmtSF',
     'SaleType',
     'BsmtFullBath',
     'MSZoning',
     'Exterior1st',
     'SalePrice']




```python
# count of variables missing values in train but not test set
train_missing.loc[train_not_test]
```




    Electrical    1
    dtype: int64




```python
# count of variables missing values in test but not train
test_missing.loc[test_not_train].drop(index=['SalePrice'])
```




    GarageArea      1
    Exterior2nd     1
    BsmtHalfBath    2
    BsmtUnfSF       1
    BsmtFinSF2      1
    BsmtFinSF1      1
    Functional      2
    Utilities       2
    GarageCars      1
    TotalBsmtSF     1
    SaleType        1
    BsmtFullBath    2
    MSZoning        4
    Exterior1st     1
    dtype: int64



Since there are so few missing values for variables which are missing values in the train set not the test set (or vice versa), we won't worry about imputing them. 

Now let's look at the distributions of variables missing in both train and test sets


```python
def plot_both_train_and_test_missing_values(hpdf):
    """plot distribution of missing train values."""
    
    copy = hpdf.data.drop(columns=['SalePrice'])
    train_missing_dist = HPDataFramePlus(data=clean.data.loc['train', :]).na_counts()
    train_missing_dist = train_missing_dist/sum(train_missing_dist)
    
    test_missing_dist = HPDataFramePlus(data=clean.data.loc['test', :]).na_counts()
    test_missing_dist = test_missing_dist/sum(test_missing_dist)
    
    both_missing_index = set(train_missing_dist.index).intersection(test_missing_dist.index)
    train_missing_dist = train_missing_dist.loc[both_missing_index]
    test_missing_dist = test_missing_dist.loc[both_missing_index]
    
    fig, _ = plt.subplots(1, 2, figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=train_missing_dist.index, y=train_missing_dist.values)
    plt.xticks(rotation=60)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=test_missing_dist.index, y=test_missing_dist.values)
    plt.xticks(rotation=60)

plot_both_train_and_test_missing_values(clean)
```


![png]({{site.baseurl}}/assets/images/process_58_0.png)


For variables missing values in both sets, the distributions are very similar, so we'll go ahead and impute these values

#### Impute small numbers of missing values by hand

Imputation of missing values using point estimates (mean, mode, etc.) is very common but arguable somewhat crude. Since there are more sophisticated methods which aren't too difficult to use, we'd like to use them. They are however, a bit more computationally expensive. Since many of our variables are only missing a few values, imputing these values by hand using point estimates will cut down on computational cost while sacrificing little.

An excellent, thorough treatment of imputation can be found in [Flexible Imputation of Missing Data](https://stefvanbuuren.name/fimd/) by Stef Van Buren.


```python
# Impute variables with <= 4 missing values. Use mode for categoricals, median for quantitatives
clean.data = clean.hand_impute()
```


```python
# missing value counts again
clean.na_counts().drop(index=['SalePrice'])
```




    LotFrontage    485
    MasVnrType      24
    MasVnrArea      23
    GarageType     157
    GarageYrBlt    159
    dtype: int64



#### Impute missing categorical values with `XGBClassifier`

Some methods for imputing missing categorical data are more common, e.g. multinomial classification, but any classifier will do. Given time and the inclination, one could explore different classifiers and try to estimate their imputation accuracy (e.g. by cross-validation on data with no missing values) but we won't do that here. Since `xgboost` classifier often performs very well with defaults, we'll use it to impute `MasVnrType` and `GarageType`.


```python
# impute missing categorical values with XGBClassifier
clean.data = clean.impute_cats(response='SalePrice')
```


```python
# missing value counts again
clean.na_counts().drop(index=['SalePrice'])
```




    LotFrontage    485
    MasVnrArea      23
    GarageYrBlt    159
    dtype: int64



#### Impute missing quantitative values with MICE and PMM

[Multiple Imputation with Chained Equations (MICE)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/) is principled method of imputing missing data. It can be combined with [Predictive Mean Matching (PMM)](http://stefvanbuuren.name/fimd/sec-pmm.html) to yield a powerful implementation method. One can find these methods implemented in Python in `statsmodels.imputation.mice`.


```python
# impute missing quantitative values with MICE and PMM
clean.data = clean.impute_quants(response='SalePrice')
```


```python
# missing value counts again
clean.na_counts().drop(index=['SalePrice'])
```




    Series([], dtype: int64)



### Enforce dtypes

On top of our `col_kinds` dictionary, we'll use pandas dtypes to track categorical, ordinal, and quantitative variables


```python
cats, ords, quants = (clean.col_kinds['cat'], clean.col_kinds['ord'],
                          clean.col_kinds['quant'])
clean.data.loc[:, cats] = clean.data.loc[:, cats].astype('category')
clean.data.loc[:, ords] = clean.data.loc[:, ords].astype('int64')
clean.data.loc[:, 'MSSubClass'] = clean.data['MSSubClass'].astype(
                                      'category')
clean.data.loc[:, quants] = clean.data.loc[:, quants].astype('float64')
clean.data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 2917 entries, (train, 1) to (test, 2919)
    Data columns (total 78 columns):
    MSSubClass       2917 non-null category
    MSZoning         2917 non-null category
    LotFrontage      2917 non-null float64
    LotArea          2917 non-null float64
    Street           2917 non-null category
    LotShape         2917 non-null int64
    LandContour      2917 non-null category
    Utilities        2917 non-null int64
    LotConfig        2917 non-null category
    LandSlope        2917 non-null int64
    Neighborhood     2917 non-null category
    Condition1       2917 non-null category
    Condition2       2917 non-null category
    BldgType         2917 non-null category
    HouseStyle       2917 non-null category
    OverallQual      2917 non-null int64
    OverallCond      2917 non-null int64
    YearBuilt        2917 non-null float64
    YearRemodAdd     2917 non-null float64
    RoofStyle        2917 non-null category
    RoofMatl         2917 non-null category
    Exterior1st      2917 non-null category
    Exterior2nd      2917 non-null category
    MasVnrType       2917 non-null category
    MasVnrArea       2917 non-null float64
    ExterQual        2917 non-null int64
    ExterCond        2917 non-null int64
    Foundation       2917 non-null category
    BsmtQual         2917 non-null int64
    BsmtCond         2917 non-null int64
    BsmtExposure     2917 non-null int64
    BsmtFinType1     2917 non-null int64
    BsmtFinSF1       2917 non-null float64
    BsmtFinType2     2917 non-null int64
    BsmtFinSF2       2917 non-null float64
    BsmtUnfSF        2917 non-null float64
    TotalBsmtSF      2917 non-null float64
    Heating          2917 non-null category
    HeatingQC        2917 non-null int64
    CentralAir       2917 non-null category
    Electrical       2917 non-null category
    1stFlrSF         2917 non-null float64
    2ndFlrSF         2917 non-null float64
    LowQualFinSF     2917 non-null float64
    GrLivArea        2917 non-null float64
    BsmtFullBath     2917 non-null int64
    BsmtHalfBath     2917 non-null int64
    FullBath         2917 non-null int64
    HalfBath         2917 non-null int64
    BedroomAbvGr     2917 non-null int64
    KitchenAbvGr     2917 non-null int64
    KitchenQual      2917 non-null int64
    TotRmsAbvGrd     2917 non-null int64
    Functional       2917 non-null int64
    Fireplaces       2917 non-null int64
    FireplaceQu      2917 non-null int64
    GarageType       2917 non-null category
    GarageYrBlt      2917 non-null float64
    GarageFinish     2917 non-null int64
    GarageCars       2917 non-null int64
    GarageArea       2917 non-null float64
    GarageQual       2917 non-null int64
    GarageCond       2917 non-null int64
    PavedDrive       2917 non-null int64
    WoodDeckSF       2917 non-null float64
    OpenPorchSF      2917 non-null float64
    EnclosedPorch    2917 non-null float64
    3SsnPorch        2917 non-null float64
    ScreenPorch      2917 non-null float64
    PoolArea         2917 non-null float64
    PoolQC           2917 non-null int64
    Fence            2917 non-null int64
    MiscVal          2917 non-null float64
    MoSold           2917 non-null int64
    YrSold           2917 non-null int64
    SaleType         2917 non-null category
    SaleCondition    2917 non-null category
    SalePrice        1458 non-null float64
    dtypes: category(22), float64(23), int64(33)
    memory usage: 1.4+ MB


## Save processed data

Finally we'll save our datasets to disk. This will result in two files

- `orig.csv` - Original train and test data combined in a single dataset, without any modification
- `clean.csv` - Cleaned dataset, with problematic variables and observations dropped and missing values imputed


```python
hp_data = DataPlus({'orig': orig, 'clean': clean})
data_dir = '../data'
hp_data.save_dfs(save_dir=data_dir)
```
