---
layout: page
title: Exploratory analysis
---

In a previous notebook, we [processed and cleaned]({{site.baseurl}}/process/) the Ames housing dataset. In this notebook, we focus on exploring the variables and the relationships among them. In a later notebook we'll 
[model]({{site.baseurl}}/model/) and predict sale prices.

## Contents

- [Setup](#setup)

- [Load and inspect data](#load-and-inspect-data)

- [The response variable `SalePrice`](#the-response-variable-saleprice)
	- [Testing log-normality](#testing-log-normality)
		- [QQ-plot](#qq-plot)
		- [Kolmogorov-Smirnov test](#kolmogorov-smirnov-test)

- [Categorical variables](#categorical-variables)
  - [Distributions of categorical variables](#distributions-of-categorical-variables)
	- [Relationships among categorical variables](#relationships-among-categorical-variables)
	- [Relationships between categoricals and `SalePrice`](#relationships-between-categoricals-and-saleprice)

- [Ordinal variables](#ordinal-variables)
	- [Distributions of ordinal variables](#distributions-of-ordinal-variables)
	- [Relationships among ordinal variables](#relationships-among-ordinal-variables)
	- [Relationships between ordinals and `SalePrice`](#relationships-between-ordinals-and-saleprice)

- [Quantitative variables](#quantitative-variables)
  - [Distributions of quantitative variables](#distributions-of-quantitative-variables)
	- [Relationships among quantitative variables](#relationships-among-quantitative-variables)
	- [Relationships between quantitatives and `SalePrice`](#relationships-between-quantitatives-and-saleprice)

## Setup


```python
# standard imports
%matplotlib inline
import matplotlib.pyplot as plt
import warnings
import scipy.stats as ss
import sys
import os

# add parent directory for importing custom classes
pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(pardir)

# add root site-packages directory to workaround pyitlib pip install issue
sys.path.append('/Users/home/anaconda3/lib/python3.7/site-packages')

# custom classes
from codes.process import DataDescription
from codes.explore import *

# notebook settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-white')
sns.set_style('white')
```

## Load and inspect data


```python
data_dir = '../data'
file_names = ['orig.csv', 'clean.csv']
hp_data = load_datasets(data_dir, file_names)
orig, clean = (hp_data.dfs['orig'], hp_data.dfs['clean'])                                    
```

We have 2 versions of the dataset here (created in [a previous notebook](preprocess.ipynb/#Preprocessing-the-Ames-housing-dataset))

- `orig` is the original dataset with no preprocessing
- `clean` is the preprocessed dataset, with problematic variables and observations dropped and missing values imputed

. In this notebook we'll primarily be working with the cleaned dataset


```python
clean.data.head()
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
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
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
      <td>8450.0</td>
      <td>Pave</td>
      <td>0</td>
      <td>Lvl</td>
      <td>3</td>
      <td>Inside</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
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
      <td>9600.0</td>
      <td>Pave</td>
      <td>0</td>
      <td>Lvl</td>
      <td>3</td>
      <td>FR2</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
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
      <td>11250.0</td>
      <td>Pave</td>
      <td>1</td>
      <td>Lvl</td>
      <td>3</td>
      <td>Inside</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
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
      <td>9550.0</td>
      <td>Pave</td>
      <td>1</td>
      <td>Lvl</td>
      <td>3</td>
      <td>Corner</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
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
      <td>14260.0</td>
      <td>Pave</td>
      <td>1</td>
      <td>Lvl</td>
      <td>3</td>
      <td>FR2</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 78 columns</p>
</div>




```python
clean.data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 2916 entries, (train, 1) to (test, 2919)
    Data columns (total 78 columns):
    MSSubClass       2916 non-null category
    MSZoning         2916 non-null category
    LotFrontage      2916 non-null float64
    LotArea          2916 non-null float64
    Street           2916 non-null category
    LotShape         2916 non-null int64
    LandContour      2916 non-null category
    Utilities        2916 non-null int64
    LotConfig        2916 non-null category
    LandSlope        2916 non-null int64
    Neighborhood     2916 non-null category
    Condition1       2916 non-null category
    Condition2       2916 non-null category
    BldgType         2916 non-null category
    HouseStyle       2916 non-null category
    OverallQual      2916 non-null int64
    OverallCond      2916 non-null int64
    YearBuilt        2916 non-null float64
    YearRemodAdd     2916 non-null float64
    RoofStyle        2916 non-null category
    RoofMatl         2916 non-null category
    Exterior1st      2916 non-null category
    Exterior2nd      2916 non-null category
    MasVnrType       2916 non-null category
    MasVnrArea       2916 non-null float64
    ExterQual        2916 non-null int64
    ExterCond        2916 non-null int64
    Foundation       2916 non-null category
    BsmtQual         2916 non-null int64
    BsmtCond         2916 non-null int64
    BsmtExposure     2916 non-null int64
    BsmtFinType1     2916 non-null int64
    BsmtFinSF1       2916 non-null float64
    BsmtFinType2     2916 non-null int64
    BsmtFinSF2       2916 non-null float64
    BsmtUnfSF        2916 non-null float64
    TotalBsmtSF      2916 non-null float64
    Heating          2916 non-null category
    HeatingQC        2916 non-null int64
    CentralAir       2916 non-null category
    Electrical       2916 non-null category
    1stFlrSF         2916 non-null float64
    2ndFlrSF         2916 non-null float64
    LowQualFinSF     2916 non-null float64
    GrLivArea        2916 non-null float64
    BsmtFullBath     2916 non-null int64
    BsmtHalfBath     2916 non-null int64
    FullBath         2916 non-null int64
    HalfBath         2916 non-null int64
    BedroomAbvGr     2916 non-null int64
    KitchenAbvGr     2916 non-null int64
    KitchenQual      2916 non-null int64
    TotRmsAbvGrd     2916 non-null int64
    Functional       2916 non-null int64
    Fireplaces       2916 non-null int64
    FireplaceQu      2916 non-null int64
    GarageType       2916 non-null category
    GarageYrBlt      2916 non-null float64
    GarageFinish     2916 non-null int64
    GarageCars       2916 non-null int64
    GarageArea       2916 non-null float64
    GarageQual       2916 non-null int64
    GarageCond       2916 non-null int64
    PavedDrive       2916 non-null int64
    WoodDeckSF       2916 non-null float64
    OpenPorchSF      2916 non-null float64
    EnclosedPorch    2916 non-null float64
    3SsnPorch        2916 non-null float64
    ScreenPorch      2916 non-null float64
    PoolArea         2916 non-null float64
    PoolQC           2916 non-null int64
    Fence            2916 non-null int64
    MiscVal          2916 non-null float64
    MoSold           2916 non-null int64
    YrSold           2916 non-null int64
    SaleType         2916 non-null category
    SaleCondition    2916 non-null category
    SalePrice        1457 non-null float64
    dtypes: category(22), float64(23), int64(33)
    memory usage: 1.3+ MB


## The response variable `SalePrice`

First let's look at the distribution of `SalePrice`, the variable we'll later be interested in predicting.


```python
sale_price = clean.data.loc['train', :]['SalePrice']
plt.figure(figsize=(15, 6))
sns.distplot(sale_price)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c407cc0>




![png]({{site.baseurl}}/assets/images/explore_11_1.png)



```python
plt.figure(figsize=(15, 6))
sns.swarmplot(sale_price)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c580d68>




![png]({{site.baseurl}}/assets/images/explore_12_1.png)


The distribution is positively skewed, with a long right tail. There are two observations with `SalePrice` > 700000, and with a good separation from the rest of the points.


```python
# check skewness of SalePrice
sale_price.skew()
```




    1.88374941136315



### Testing log-normality

The distribution looks like it may be approximately log-normal, let's check this


```python
# distribution of log(SalePrice)
plt.figure(figsize=(15, 6))
sns.distplot(np.log(sale_price))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c733b00>




![png]({{site.baseurl}}/assets/images/explore_17_1.png)


#### QQ-plot


```python
# lognormal QQ plot
plt.figure(figsize=(8, 8))
# standard deviation of log is the shape parameter
s = np.log(sale_price).std()
lognorm = ss.lognorm(s)
ss.probplot(sale_price, dist=lognorm, plot=plt)
plt.show()
```


![png]({{site.baseurl}}/assets/images/explore_19_0.png)


The distribution appears to be approxiately log-normal, although the QQ plot shows the right tail is a bit longer than expected, and the two observations with highest `SalePrice` are much higher than expected.

#### Kolmogorov - Smirnov test

This is a [non-parametric test for comparing distributions](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test). We'll use `scipy.stats` [implementation](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html)


```python
ss.kstest(sale_price, lognorm.cdf)
```




    KstestResult(statistic=1.0, pvalue=0.0)



This test conclusively rejects the null hypothesis that the distribution of `SalePrice` is lognormal. Nevertheless, the plots indicate that log-normality is perhaps a usefull approximation. Moreover, `log(SalePrice)` may be more useful than `SalePrice` for prediction purposes, given the symmetry of its distribution.

## Categorical variables

First we look at all categorical variables, that is, all discrete variables with no ordering on the values. In our cleaned dataframe these are all the columns with `category` dtype


```python
# dataframe of categorical variables
cats = HPDataFramePlus(data=clean.data.select_dtypes('category'))
cats.data.head()
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
      <th>Street</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>...</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>Foundation</th>
      <th>Heating</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>GarageType</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
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
      <td>Pave</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>...</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>PConc</td>
      <td>GasA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Attchd</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>Pave</td>
      <td>Lvl</td>
      <td>FR2</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>...</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>CBlock</td>
      <td>GasA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Attchd</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>Pave</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>...</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>PConc</td>
      <td>GasA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Attchd</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>Pave</td>
      <td>Lvl</td>
      <td>Corner</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>...</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>BrkTil</td>
      <td>GasA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Detchd</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>Pave</td>
      <td>Lvl</td>
      <td>FR2</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>...</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>PConc</td>
      <td>GasA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Attchd</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
# print description of categorical variables
desc = DataDescription('../data/data_description.txt')
cats.desc = desc
cats.print_desc(cols=cats.data.columns)
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
    
    
    Street: Type of road access to property
    
    	 Grvl - Gravel
    	 Pave - Paved
    
    
    LandContour: Flatness of the property
    
    	 Lvl - Near Flat/Level
    	 Bnk - Banked - Quick and significant rise from street grade to building
    	 HLS - Hillside - Significant slope from side to side
    	 Low - Depression
    
    
    LotConfig: Lot configuration
    
    	 Inside - Inside lot
    	 Corner - Corner lot
    	 CulDSac - Cul-de-sac
    	 FR2 - Frontage on 2 sides of property
    	 FR3 - Frontage on 3 sides of property
    
    
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
    	 Wd - Wood Siding
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
    	 Wd - Wood Siding
    	 WdShing - Wood Shingles
    
    
    MasVnrType: Masonry veneer type
    
    	 BrkCmn - Brick Common
    	 BrkFace - Brick Face
    	 CBlock - Cinder Block
    	 None - None
    	 Stone - Stone
    
    
    Foundation: Type of foundation
    
    	 BrkTil - Brick & Tile
    	 CBlock - Cinder Block
    	 PConc - Poured Contrete
    	 Slab - Slab
    	 Stone - Stone
    	 Wood - Wood
    
    
    Heating: Type of heating
    
    	 Floor - Floor Furnace
    	 GasA - Gas forced warm air furnace
    	 GasW - Gas hot water or steam heat
    	 Grav - Gravity furnace
    	 OthW - Hot water or steam heat other than gas
    	 Wall - Wall furnace
    
    
    CentralAir: Central air conditioning
    
    	 N - No
    	 Y - Yes
    
    
    Electrical: Electrical system
    
    	 SBrkr - Standard Circuit Breakers & Romex
    	 FuseA - Fuse Box over 60 AMP and all Romex wiring (Average)
    	 FuseF - 60 AMP Fuse Box and mostly Romex wiring (Fair)
    	 FuseP - 60 AMP Fuse Box and mostly knob & tube wiring (poor)
    	 Mix - Mixed
    
    
    GarageType: Garage location
    
    	 2Types - More than one type of garage
    	 Attchd - Attached to home
    	 Basment - Basement Garage
    	 BuiltIn - Built-In (Garage part of house - typically has room above garage)
    	 CarPort - Car Port
    	 Detchd - Detached from home
    	 NA - No Garage
    
    
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
    
    


### Distributions of categorical variables


```python
# plot distributions of categorical variables
plot_discrete_dists(nrows=8, ncols=3, data=cats.data, figsize=(15, 30))
```


![png]({{site.baseurl}}/assets/images/explore_30_0.png)


Some of these variables have highly unbalanced distributions. We'll look for the most extremely unbalanced


```python
# print distributions of categorical variables with more 90% concentration at single value
unbal_cat_cols = print_unbal_dists(data=cats.data, bal_threshold=0.9)
```

    
    Pave    0.995885
    Grvl    0.004115
    Name: Street, dtype: float64
    
    Norm      0.990055
    Feedr     0.004458
    Artery    0.001715
    PosA      0.001372
    PosN      0.001029
    RRNn      0.000686
    RRAn      0.000343
    RRAe      0.000343
    Name: Condition2, dtype: float64
    
    CompShg    0.985940
    Tar&Grv    0.007545
    WdShake    0.003086
    WdShngl    0.002401
    Roll       0.000343
    Metal      0.000343
    Membran    0.000343
    Name: RoofMatl, dtype: float64
    
    GasA     0.984568
    GasW     0.009259
    Grav     0.003086
    Wall     0.002058
    OthW     0.000686
    Floor    0.000343
    Name: Heating, dtype: float64
    
    Y    0.932785
    N    0.067215
    Name: CentralAir, dtype: float64
    
    SBrkr    0.915295
    FuseA    0.064472
    FuseF    0.017147
    FuseP    0.002743
    Mix      0.000343
    Name: Electrical, dtype: float64


### Relationships among categorical variables

{% katexmm %}

One often speaks loosely of "correlation" among variables to refer to statistical dependence. There are various measures of dependence, but here we rely on an information theoretic measure known as the [variation of information](https://en.wikipedia.org/wiki/Variation_of_information). We discuss this measure briefly

The function

$$d(X, Y) = H(X, Y) - I(X, Y)$$

where $H(X, Y)$ is the joint entropy and $I(X, Y)$ the mutual information, [defines a metric](https://arxiv.org/pdf/q-bio/0311039.pdf) on a set of discrete random variables. Note that

$$d(X, Y) = H(X|Y) + H(Y|X)$$

which is sometimes called the "variation of information". One can normalize to get a standardized variation of information

$$D(X, Y) = \frac{d(X, Y)}{H(X, Y)} = 1 - \frac{I(X, Y)}{H(X, Y)} $$

i.e. $D(X, Y) \in [0, 1]$. Since $D$ is a metric, $D(X, Y) = 0$ iff $X = Y$ Furthermore, $D(X, Y) = 1$ if and only if $I(X, Y) = 0$ if and only if $X, Y$ are independendent. So we can take $D(X, Y)$ as a "dependence distance". The closer a variable $Y$ is to $X$, the more it depends on $X$. 

Of course, we don't know the true distributions of the random variables in this data set, but the sample size is large enough that the sample distributions should be a good approximation. 

We'll look at the dependence distance among variables with feature selection in mind, namely the possibility of removing redundant variables.


```python
# Get dataframe of dependence distances of categorical variables
cats_data_num = num_enc(data=cats.data)
cats_D_dep_df = D_dep(data=cats_data_num)
```


```python
# plot all dependence distances
plot_D_dep(cats_D_dep_df, figsize=(15, 10))
```


![png]({{site.baseurl}}/assets/images/explore_37_0.png)



```python
# plot dependence distances less than 0.8
plot_low_D_dep(D_dep_df=cats_D_dep_df, D_threshold=0.8, figsize=(13, 8))
```


![png]({{site.baseurl}}/assets/images/explore_38_0.png)



```python
# rank categorical variables by dependence distance
rank_pairs_by_D(D_dep_df=cats_D_dep_df, D_threshold=0.8)
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
      <th>var1</th>
      <th>var2</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Exterior1st</td>
      <td>Exterior2nd</td>
      <td>0.322737</td>
    </tr>
    <tr>
      <td>2</td>
      <td>MSSubClass</td>
      <td>HouseStyle</td>
      <td>0.472661</td>
    </tr>
    <tr>
      <td>3</td>
      <td>SaleType</td>
      <td>SaleCondition</td>
      <td>0.667950</td>
    </tr>
    <tr>
      <td>4</td>
      <td>MSSubClass</td>
      <td>BldgType</td>
      <td>0.714236</td>
    </tr>
  </tbody>
</table>
</div>


 Notable pairs of distinct variables with low dependence distance are
 
- `Exterior1st` and `Exterior2nd` have the lowest dependence distance ($D \approx 0.322$). Their distributions are very similar and they have the same values. It probably makes more sense to think of them as close to identically distributed. 
- `MSSubclass` and `HouseStyle` have the next lowest ($D \approx 0.47$). Inspecting their descriptions above we see that they have very similar categories, so they are measuring very similar things. `BldgType` and `MSSubclass` ($D \approx 0.71$) are similar. 
- `MSSubclass` and `Neighborhood` ($D \approx 0.84$) are perhaps the first interesting pair in that they are measuring different things. We can imagine that the association between these two variables is somewhat strong -- it makes sense that the size/age/type of house would be related to the neighborhood. Similarly, `Exterior1st`, `Exterior2nd`, `MSZoning`, `Foundation` also have strong associations with `Neighborhood`.
- `SaleCondition` and `SaleType` ($D \approx 0.67$) are also unsurprisingly associated. 

{% endkatexmm %}

### Relationships between categoricals and `SalePrice`

Given that `SalePrice` seemed to be [well-approximated](#Testing-log-normality) by a log-normal distribution, we'll measure dependence with `log_SalePrice`.


```python
cats_data_num['log_SalePrice'] = np.log(clean.data['SalePrice'])
cats_data_num['log_SalePrice']
```




           Id  
    train  1       12.247694
           2       12.109011
           3       12.317167
           4       11.849398
           5       12.429216
                     ...    
    test   2915          NaN
           2916          NaN
           2917          NaN
           2918          NaN
           2919          NaN
    Name: log_SalePrice, Length: 2916, dtype: float64



To visualize the relationship between the categorical variables and the response, we'll look at the distributions of `log_SalePrice` over the values of the variables.


```python
# violin plots of categorical variables vs. response
plot_violin_plots(nrows=8, ncols=3, data=cats_data_num, response='log_SalePrice', figsize=(15, 30))
```


![png]({{site.baseurl}}/assets/images/explore_45_0.png)


Note that horizontal lines for variable values in the violin plots indicate less than 5 observations having that value

From these plots, it's difficult to determine with accuracy for which variables the distribution of `log_SalePrice` doesn't seem to vary greatly across values (and hence are of low dependence and thus low predictive value). The dependence distance between the variables and `log_SalePrice` will provide additional information.


```python
# rank categorical variables by dependence distance from response
D_dep_response(cats_data_num, 'log_SalePrice').sort_values(by='D').T
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
      <th>Neighborhood</th>
      <th>MSSubClass</th>
      <th>Exterior2nd</th>
      <th>Exterior1st</th>
      <th>HouseStyle</th>
      <th>Foundation</th>
      <th>GarageType</th>
      <th>MasVnrType</th>
      <th>SaleCondition</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>Condition1</th>
      <th>BldgType</th>
      <th>RoofStyle</th>
      <th>LandContour</th>
      <th>Electrical</th>
      <th>CentralAir</th>
      <th>Heating</th>
      <th>RoofMatl</th>
      <th>Condition2</th>
      <th>Street</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>D</td>
      <td>0.713181</td>
      <td>0.813289</td>
      <td>0.831957</td>
      <td>0.838796</td>
      <td>0.894477</td>
      <td>0.901514</td>
      <td>0.919312</td>
      <td>0.924422</td>
      <td>0.926004</td>
      <td>0.929782</td>
      <td>...</td>
      <td>0.937908</td>
      <td>0.938181</td>
      <td>0.947682</td>
      <td>0.956498</td>
      <td>0.966163</td>
      <td>0.973683</td>
      <td>0.990074</td>
      <td>0.990213</td>
      <td>0.991566</td>
      <td>0.996507</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 22 columns</p>
</div>



The lower the dependence distance here, the better assocation with the response, hence the better the potential predictive value.

In particular, given how unbalanced their distributions are, it's perhaps not too surprising to see `RoofStyle`, `LandContour`, `Electrical` and `CentralAir` all have such high dependence distance, 

## Ordinal variables

Now we'll investigate ordinal variables, that is discrete variables with an ordering. In our cleaned dataframe these are variables with `int64` dtype


```python
# dataframe of ordinal variables
ords = HPDataFramePlus(data=clean.data.select_dtypes('int64'))
ords.data.head()
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
      <th>LotShape</th>
      <th>Utilities</th>
      <th>LandSlope</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>...</th>
      <th>FireplaceQu</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MoSold</th>
      <th>YrSold</th>
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
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>7</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>7</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>8</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
ords.data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 2916 entries, (train, 1) to (test, 2919)
    Data columns (total 33 columns):
    LotShape        2916 non-null int64
    Utilities       2916 non-null int64
    LandSlope       2916 non-null int64
    OverallQual     2916 non-null int64
    OverallCond     2916 non-null int64
    ExterQual       2916 non-null int64
    ExterCond       2916 non-null int64
    BsmtQual        2916 non-null int64
    BsmtCond        2916 non-null int64
    BsmtExposure    2916 non-null int64
    BsmtFinType1    2916 non-null int64
    BsmtFinType2    2916 non-null int64
    HeatingQC       2916 non-null int64
    BsmtFullBath    2916 non-null int64
    BsmtHalfBath    2916 non-null int64
    FullBath        2916 non-null int64
    HalfBath        2916 non-null int64
    BedroomAbvGr    2916 non-null int64
    KitchenAbvGr    2916 non-null int64
    KitchenQual     2916 non-null int64
    TotRmsAbvGrd    2916 non-null int64
    Functional      2916 non-null int64
    Fireplaces      2916 non-null int64
    FireplaceQu     2916 non-null int64
    GarageFinish    2916 non-null int64
    GarageCars      2916 non-null int64
    GarageQual      2916 non-null int64
    GarageCond      2916 non-null int64
    PavedDrive      2916 non-null int64
    PoolQC          2916 non-null int64
    Fence           2916 non-null int64
    MoSold          2916 non-null int64
    YrSold          2916 non-null int64
    dtypes: int64(33)
    memory usage: 783.3+ KB


We'll print the description of all variables, however note that the print description contains the original value for the variables, while the cleaned dataframe `clean` contains the [numerically encoded values](preprocess.ipynb/#Encode-variables)


```python
# print description of ordinal variables
ords.desc = desc
ords.print_desc(cols=ords.data.columns)
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
    
    
    


### Distributions of ordinal variables


```python
# plot distributions of ordinal variables
plot_discrete_dists(nrows=11, ncols=3, data=ords.data, figsize=(15, 30))
```


![png]({{site.baseurl}}/assets/images/explore_57_0.png)



```python
# look at most unbalanced distributions
unbal_ord_cols = print_unbal_dists(data=ords.data, bal_threshold=0.9)
```

    
    3    0.999657
    1    0.000343
    Name: Utilities, dtype: float64
    
    0    0.951989
    1    0.042524
    2    0.005487
    Name: LandSlope, dtype: float64
    
    0    0.939986
    1    0.058642
    2    0.001372
    Name: BsmtHalfBath, dtype: float64
    
    1    0.954047
    2    0.044239
    0    0.001029
    3    0.000686
    Name: KitchenAbvGr, dtype: float64
    
    6    0.931756
    3    0.024005
    5    0.021948
    2    0.012003
    4    0.006516
    1    0.003086
    0    0.000686
    Name: Functional, dtype: float64
    
    3    0.909122
    0    0.054527
    2    0.025377
    4    0.005144
    1    0.004801
    5    0.001029
    Name: GarageCond, dtype: float64
    
    2    0.904664
    0    0.074074
    1    0.021262
    Name: PavedDrive, dtype: float64
    
    0    0.996914
    4    0.001372
    3    0.001029
    1    0.000686
    Name: PoolQC, dtype: float64


### Relationships among ordinal variables


```python
# get dataframe of dependence distances of ordinal variables
ords_D_dep_df = D_dep(ords.data)
```


```python
# plot all dependence distances
plot_D_dep(D_dep_df=ords_D_dep_df, figsize=(15, 10))
```


![png]({{site.baseurl}}/assets/images/explore_61_0.png)



```python
# plot lower dependence distances of ordinal variables
plot_low_D_dep(D_dep_df=ords_D_dep_df, D_threshold=0.8, figsize=(13, 8))
```


![png]({{site.baseurl}}/assets/images/explore_62_0.png)



```python
# rank ordinals by low dependence distance
rank_pairs_by_D(D_dep_df=ords_D_dep_df, D_threshold=0.8)
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
      <th>var1</th>
      <th>var2</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Fireplaces</td>
      <td>FireplaceQu</td>
      <td>0.528211</td>
    </tr>
    <tr>
      <td>2</td>
      <td>GarageQual</td>
      <td>GarageCond</td>
      <td>0.542600</td>
    </tr>
    <tr>
      <td>3</td>
      <td>ExterQual</td>
      <td>KitchenQual</td>
      <td>0.760176</td>
    </tr>
  </tbody>
</table>
</div>

{% katexmm %}

 Notable pairs of distinct ordinal variables with low dependence distance are
 
- `Fireplaces` and `FireplaceQu` have the lowest dependence distance ($D \approx 0.53$). This is somewhat interesting, in that the quantities these variables are measuring are distinct (albeit related).
- `GarageQual` and `GarageCond` have the next lowest ($D \approx 0.54$). Inspecting their descriptions above we see that they have very similar categories, so they are measuring very similar things. There is ostensibly a distinction between the quality of the garage and its condition, however.
- Pairs of garage variables display relatively low dependence distance, as do pairs of basement variables and quality variables.

{% endkatexmm %}

### Relationships between ordinals and `SalePrice`


```python
# add log_SalePrice to ordinal dataframe
ords.data['log_SalePrice'] = np.log(clean.data['SalePrice'])
ords.data['log_SalePrice']
```




           Id  
    train  1       12.247694
           2       12.109011
           3       12.317167
           4       11.849398
           5       12.429216
                     ...    
    test   2915          NaN
           2916          NaN
           2917          NaN
           2918          NaN
           2919          NaN
    Name: log_SalePrice, Length: 2916, dtype: float64




```python
# violin plots of ordinals
plot_violin_plots(11, 3, ords.data, 'log_SalePrice', figsize=(15, 30))
```


![png]({{site.baseurl}}/assets/images/explore_67_0.png)



```python
# plot dependence distance with log_SalePrice
D_dep_response(ords.data, 'log_SalePrice').sort_values(by='D').T
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
      <th>MoSold</th>
      <th>OverallQual</th>
      <th>TotRmsAbvGrd</th>
      <th>BsmtFinType1</th>
      <th>YrSold</th>
      <th>GarageCars</th>
      <th>BsmtQual</th>
      <th>GarageFinish</th>
      <th>FireplaceQu</th>
      <th>OverallCond</th>
      <th>...</th>
      <th>GarageQual</th>
      <th>ExterCond</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>Functional</th>
      <th>BsmtHalfBath</th>
      <th>LandSlope</th>
      <th>KitchenAbvGr</th>
      <th>PoolQC</th>
      <th>Utilities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>D</td>
      <td>0.795577</td>
      <td>0.821511</td>
      <td>0.83499</td>
      <td>0.859118</td>
      <td>0.877148</td>
      <td>0.879704</td>
      <td>0.879858</td>
      <td>0.886311</td>
      <td>0.886812</td>
      <td>0.890191</td>
      <td>...</td>
      <td>0.951855</td>
      <td>0.957094</td>
      <td>0.959353</td>
      <td>0.962894</td>
      <td>0.965643</td>
      <td>0.978447</td>
      <td>0.979514</td>
      <td>0.981345</td>
      <td>0.997317</td>
      <td>0.999801</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 33 columns</p>
</div>



Again variables with unbalanced distributions (e.g. `PoolQc`, `Utilities`) tend to have high dependence distance, as do variables for which the distribution of `log_SalePrice` doesn't differ much across their classes (e.g. `BsmtHalfBath`, `PavedDrive`, `LandSlope`).

That `OverallQual` has high dependence with `SalePrice` isn't surprising, but perhaps `MoSold` having the lowest is.

#### Rank correlation hypothesis tests

{% katexmm %}

One way of testing statistical dependence between ordered varialbes is using [rank correlations](https://en.wikipedia.org/wiki/Rank_correlation). Since they're relatively straightforward to calculate, we calculate three common ones and compare. We'll look at [Pearson's $\rho$](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient), [Spearman's $r_s$](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient), and [Kendall's $\tau$](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)


```python
# rank correlation results as dataframes
rho_df = rank_hyp_test(ords, 'rho', ss.pearsonr)
r_s_df = rank_hyp_test(ords, 'r_s', ss.spearmanr)
tau_df = rank_hyp_test(ords, 'tau', ss.kendalltau)
rank_hyp_test_dfs = {'rho': rho_df, 'r_s': r_s_df, 'tau': tau_df}

# rank and sort by p-value of Pearson's rho
get_rank_corr_df(rank_hyp_test_dfs).drop(columns=['rho', 'r_s', 'tau']).sort_values(by='rho_p_val_rank')
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
      <th>rho_p_val</th>
      <th>rho_p_val_rank</th>
      <th>r_s_p_val</th>
      <th>r_s_p_val_rank</th>
      <th>tau_p_val</th>
      <th>tau_p_val_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>OverallQual</td>
      <td>0.000000e+00</td>
      <td>1</td>
      <td>0.000000e+00</td>
      <td>1</td>
      <td>5.929359e-270</td>
      <td>1</td>
    </tr>
    <tr>
      <td>ExterQual</td>
      <td>7.761033e-201</td>
      <td>2</td>
      <td>2.040959e-203</td>
      <td>3</td>
      <td>1.272156e-159</td>
      <td>4</td>
    </tr>
    <tr>
      <td>GarageCars</td>
      <td>3.307683e-199</td>
      <td>3</td>
      <td>2.382463e-207</td>
      <td>2</td>
      <td>6.327182e-176</td>
      <td>2</td>
    </tr>
    <tr>
      <td>KitchenQual</td>
      <td>2.324509e-190</td>
      <td>4</td>
      <td>3.122308e-193</td>
      <td>5</td>
      <td>1.456887e-158</td>
      <td>5</td>
    </tr>
    <tr>
      <td>BsmtQual</td>
      <td>5.427313e-175</td>
      <td>5</td>
      <td>2.488211e-197</td>
      <td>4</td>
      <td>1.250445e-164</td>
      <td>3</td>
    </tr>
    <tr>
      <td>GarageFinish</td>
      <td>2.620057e-146</td>
      <td>6</td>
      <td>9.382754e-165</td>
      <td>7</td>
      <td>2.217914e-140</td>
      <td>6</td>
    </tr>
    <tr>
      <td>FullBath</td>
      <td>1.759447e-141</td>
      <td>7</td>
      <td>3.253667e-167</td>
      <td>6</td>
      <td>1.117470e-133</td>
      <td>7</td>
    </tr>
    <tr>
      <td>FireplaceQu</td>
      <td>3.528296e-114</td>
      <td>8</td>
      <td>7.777438e-110</td>
      <td>8</td>
      <td>1.384314e-99</td>
      <td>9</td>
    </tr>
    <tr>
      <td>TotRmsAbvGrd</td>
      <td>3.524836e-110</td>
      <td>9</td>
      <td>4.199477e-108</td>
      <td>9</td>
      <td>5.527766e-104</td>
      <td>8</td>
    </tr>
    <tr>
      <td>Fireplaces</td>
      <td>2.049485e-89</td>
      <td>10</td>
      <td>2.189811e-101</td>
      <td>10</td>
      <td>5.443444e-88</td>
      <td>10</td>
    </tr>
    <tr>
      <td>HeatingQC</td>
      <td>2.503143e-82</td>
      <td>11</td>
      <td>2.833473e-89</td>
      <td>11</td>
      <td>5.700439e-81</td>
      <td>11</td>
    </tr>
    <tr>
      <td>GarageQual</td>
      <td>1.143613e-46</td>
      <td>12</td>
      <td>1.501771e-43</td>
      <td>13</td>
      <td>2.160589e-41</td>
      <td>13</td>
    </tr>
    <tr>
      <td>BsmtExposure</td>
      <td>3.598521e-45</td>
      <td>13</td>
      <td>1.337016e-41</td>
      <td>15</td>
      <td>1.843618e-40</td>
      <td>14</td>
    </tr>
    <tr>
      <td>GarageCond</td>
      <td>5.806508e-45</td>
      <td>14</td>
      <td>1.512574e-40</td>
      <td>16</td>
      <td>2.197488e-38</td>
      <td>16</td>
    </tr>
    <tr>
      <td>BsmtFinType1</td>
      <td>1.544276e-39</td>
      <td>15</td>
      <td>2.791158e-46</td>
      <td>12</td>
      <td>2.122343e-46</td>
      <td>12</td>
    </tr>
    <tr>
      <td>HalfBath</td>
      <td>6.573728e-35</td>
      <td>16</td>
      <td>9.576207e-42</td>
      <td>14</td>
      <td>3.530858e-39</td>
      <td>15</td>
    </tr>
    <tr>
      <td>PavedDrive</td>
      <td>1.174245e-32</td>
      <td>17</td>
      <td>9.055822e-28</td>
      <td>18</td>
      <td>6.292790e-27</td>
      <td>18</td>
    </tr>
    <tr>
      <td>LotShape</td>
      <td>3.682206e-29</td>
      <td>18</td>
      <td>3.397766e-36</td>
      <td>17</td>
      <td>1.363538e-34</td>
      <td>17</td>
    </tr>
    <tr>
      <td>BsmtCond</td>
      <td>1.302478e-26</td>
      <td>19</td>
      <td>1.100259e-25</td>
      <td>19</td>
      <td>4.811484e-25</td>
      <td>19</td>
    </tr>
    <tr>
      <td>BsmtFullBath</td>
      <td>5.765714e-20</td>
      <td>20</td>
      <td>4.174597e-18</td>
      <td>21</td>
      <td>1.040257e-17</td>
      <td>21</td>
    </tr>
    <tr>
      <td>BedroomAbvGr</td>
      <td>5.553622e-16</td>
      <td>21</td>
      <td>6.069682e-20</td>
      <td>20</td>
      <td>2.027010e-20</td>
      <td>20</td>
    </tr>
    <tr>
      <td>KitchenAbvGr</td>
      <td>1.568173e-08</td>
      <td>22</td>
      <td>2.543739e-10</td>
      <td>23</td>
      <td>3.235206e-10</td>
      <td>23</td>
    </tr>
    <tr>
      <td>Functional</td>
      <td>1.437417e-07</td>
      <td>23</td>
      <td>9.116464e-08</td>
      <td>24</td>
      <td>9.253383e-08</td>
      <td>24</td>
    </tr>
    <tr>
      <td>Fence</td>
      <td>2.386137e-05</td>
      <td>24</td>
      <td>4.477501e-13</td>
      <td>22</td>
      <td>1.551382e-12</td>
      <td>22</td>
    </tr>
    <tr>
      <td>PoolQC</td>
      <td>1.030024e-03</td>
      <td>25</td>
      <td>1.490013e-02</td>
      <td>27</td>
      <td>1.492559e-02</td>
      <td>27</td>
    </tr>
    <tr>
      <td>MoSold</td>
      <td>2.677357e-02</td>
      <td>26</td>
      <td>7.253421e-03</td>
      <td>26</td>
      <td>6.503260e-03</td>
      <td>26</td>
    </tr>
    <tr>
      <td>ExterCond</td>
      <td>5.869885e-02</td>
      <td>27</td>
      <td>6.460164e-01</td>
      <td>32</td>
      <td>6.635896e-01</td>
      <td>33</td>
    </tr>
    <tr>
      <td>YrSold</td>
      <td>1.550808e-01</td>
      <td>28</td>
      <td>2.543787e-01</td>
      <td>30</td>
      <td>2.573829e-01</td>
      <td>30</td>
    </tr>
    <tr>
      <td>OverallCond</td>
      <td>1.567476e-01</td>
      <td>29</td>
      <td>6.717184e-07</td>
      <td>25</td>
      <td>1.787177e-07</td>
      <td>25</td>
    </tr>
    <tr>
      <td>LandSlope</td>
      <td>1.671482e-01</td>
      <td>30</td>
      <td>7.277014e-02</td>
      <td>28</td>
      <td>7.323031e-02</td>
      <td>28</td>
    </tr>
    <tr>
      <td>BsmtFinType2</td>
      <td>5.861356e-01</td>
      <td>31</td>
      <td>1.246273e-01</td>
      <td>29</td>
      <td>1.412702e-01</td>
      <td>29</td>
    </tr>
    <tr>
      <td>Utilities</td>
      <td>6.304159e-01</td>
      <td>32</td>
      <td>5.249597e-01</td>
      <td>31</td>
      <td>5.247749e-01</td>
      <td>31</td>
    </tr>
    <tr>
      <td>BsmtHalfBath</td>
      <td>8.503143e-01</td>
      <td>33</td>
      <td>6.522158e-01</td>
      <td>33</td>
      <td>6.520497e-01</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>



There is more or less good agreement of $p$-value rankings among the rank correlation hypothesis tests. In particular for a 95% significance level all three fail to reject the null for `MoSold`, `ExterCond`, `OverallCond`, `LandSlope`, `BsmtFinType2`, `Utilities` and `BsmtHalfBath`. Applying a stricter value of 99.9% significance, all three reject `PoolQC` as well.

It's important to recognize that rank correlation tests are measures of monotonicity (the tendency of variables to increase together and decrease together). They may fail to detect non-linear relationships that are not monotonic. In our particular case, `MoSold` had the highest statistical dependence with `log_SalePrice` among ordinal variables, but all three rank correlation tests reject a relationship between the two at 95% significance.

{% endkatexmm %}

## Quantitative variables

Finally we'll consider the quantitative variables, that is the continuous variables. In our cleaned dataframe these are the variables with `float64` dtype.


```python
# dataframe of quantitative variables
quants = HPDataFramePlus(data=clean.data.select_dtypes('float64').drop(columns=['SalePrice']))
quants.data.head()
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
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>...</th>
      <th>GrLivArea</th>
      <th>GarageYrBlt</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
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
      <td>65.0</td>
      <td>8450.0</td>
      <td>2003.0</td>
      <td>2003.0</td>
      <td>196.0</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>150.0</td>
      <td>856.0</td>
      <td>856.0</td>
      <td>...</td>
      <td>1710.0</td>
      <td>2003.0</td>
      <td>548.0</td>
      <td>0.0</td>
      <td>61.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>80.0</td>
      <td>9600.0</td>
      <td>1976.0</td>
      <td>1976.0</td>
      <td>0.0</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>284.0</td>
      <td>1262.0</td>
      <td>1262.0</td>
      <td>...</td>
      <td>1262.0</td>
      <td>1976.0</td>
      <td>460.0</td>
      <td>298.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>68.0</td>
      <td>11250.0</td>
      <td>2001.0</td>
      <td>2002.0</td>
      <td>162.0</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>434.0</td>
      <td>920.0</td>
      <td>920.0</td>
      <td>...</td>
      <td>1786.0</td>
      <td>2001.0</td>
      <td>608.0</td>
      <td>0.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>60.0</td>
      <td>9550.0</td>
      <td>1915.0</td>
      <td>1970.0</td>
      <td>0.0</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>756.0</td>
      <td>961.0</td>
      <td>...</td>
      <td>1717.0</td>
      <td>1998.0</td>
      <td>642.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>272.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>84.0</td>
      <td>14260.0</td>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>350.0</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>490.0</td>
      <td>1145.0</td>
      <td>1145.0</td>
      <td>...</td>
      <td>2198.0</td>
      <td>2000.0</td>
      <td>836.0</td>
      <td>192.0</td>
      <td>84.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
quants.data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 2916 entries, (train, 1) to (test, 2919)
    Data columns (total 22 columns):
    LotFrontage      2916 non-null float64
    LotArea          2916 non-null float64
    YearBuilt        2916 non-null float64
    YearRemodAdd     2916 non-null float64
    MasVnrArea       2916 non-null float64
    BsmtFinSF1       2916 non-null float64
    BsmtFinSF2       2916 non-null float64
    BsmtUnfSF        2916 non-null float64
    TotalBsmtSF      2916 non-null float64
    1stFlrSF         2916 non-null float64
    2ndFlrSF         2916 non-null float64
    LowQualFinSF     2916 non-null float64
    GrLivArea        2916 non-null float64
    GarageYrBlt      2916 non-null float64
    GarageArea       2916 non-null float64
    WoodDeckSF       2916 non-null float64
    OpenPorchSF      2916 non-null float64
    EnclosedPorch    2916 non-null float64
    3SsnPorch        2916 non-null float64
    ScreenPorch      2916 non-null float64
    PoolArea         2916 non-null float64
    MiscVal          2916 non-null float64
    dtypes: float64(22)
    memory usage: 532.7+ KB



```python
# print description of quantitative variables
quants.desc = desc
quants.print_desc(cols=quants.data.columns)
```

    LotFrontage: Linear feet of street connected to property
    
    
    
    LotArea: Lot size in square feet
    
    
    
    YearBuilt: Original construction date
    
    
    
    YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
    
    
    
    MasVnrArea: Masonry veneer area in square feet
    
    
    
    BsmtFinSF1: Type 1 finished square feet
    
    
    
    BsmtFinSF2: Type 2 finished square feet
    
    
    
    BsmtUnfSF: Unfinished square feet of basement area
    
    
    
    TotalBsmtSF: Total square feet of basement area
    
    
    
    1stFlrSF: First Floor square feet
    
    
    
    2ndFlrSF: Second floor square feet
    
    
    
    LowQualFinSF: Low quality finished square feet (all floors)
    
    
    
    GrLivArea: Above grade (ground) living area square feet
    
    
    
    GarageYrBlt: Year garage was built
    
    
    
    GarageArea: Size of garage in square feet
    
    
    
    WoodDeckSF: Wood deck area in square feet
    
    
    
    OpenPorchSF: Open porch area in square feet
    
    
    
    EnclosedPorch: Enclosed porch area in square feet
    
    
    
    3SsnPorch: Three season porch area in square feet
    
    
    
    ScreenPorch: Screen porch area in square feet
    
    
    
    PoolArea: Pool area in square feet
    
    
    
    MiscVal: $Value of miscellaneous feature
    
    
    



```python
quants.data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 2916 entries, (train, 1) to (test, 2919)
    Data columns (total 22 columns):
    LotFrontage      2916 non-null float64
    LotArea          2916 non-null float64
    YearBuilt        2916 non-null float64
    YearRemodAdd     2916 non-null float64
    MasVnrArea       2916 non-null float64
    BsmtFinSF1       2916 non-null float64
    BsmtFinSF2       2916 non-null float64
    BsmtUnfSF        2916 non-null float64
    TotalBsmtSF      2916 non-null float64
    1stFlrSF         2916 non-null float64
    2ndFlrSF         2916 non-null float64
    LowQualFinSF     2916 non-null float64
    GrLivArea        2916 non-null float64
    GarageYrBlt      2916 non-null float64
    GarageArea       2916 non-null float64
    WoodDeckSF       2916 non-null float64
    OpenPorchSF      2916 non-null float64
    EnclosedPorch    2916 non-null float64
    3SsnPorch        2916 non-null float64
    ScreenPorch      2916 non-null float64
    PoolArea         2916 non-null float64
    MiscVal          2916 non-null float64
    dtypes: float64(22)
    memory usage: 532.7+ KB


```python
# plot distributions of quantitative variables
plot_cont_dists(nrows=6, ncols=4, data=quants.data, figsize=(15, 20))
```


![png]({{site.baseurl}}/assets/images/explore_81_0.png)


Most of the variables are highly positively skewed


```python
quants.data.skew()
```




    LotFrontage       1.049465
    LotArea          13.269377
    YearBuilt        -0.600024
    YearRemodAdd     -0.449893
    MasVnrArea        2.618990
    BsmtFinSF1        0.982465
    BsmtFinSF2        4.145816
    BsmtUnfSF         0.919998
    TotalBsmtSF       0.677494
    1stFlrSF          1.259407
    2ndFlrSF          0.861482
    LowQualFinSF     12.088646
    GrLivArea         1.069506
    GarageYrBlt      -0.658118
    GarageArea        0.219101
    WoodDeckSF        1.847119
    OpenPorchSF       2.533111
    EnclosedPorch     4.003630
    3SsnPorch        11.375940
    ScreenPorch       3.946335
    PoolArea         17.694707
    MiscVal          21.947201
    dtype: float64



Some of the quantitative variables appear to be multimodal. For quite a few, this is due to a large peak at zero, and for some it's clear that zero is being used as a stand-in for a null value (for example, `PoolArea` = 0 if the house has no pool). We'll look at which variables have a high peak at zero

We note that many of these variables have long right tails, so logarithmic scales may be more appropriate for these.


```python
# plot distributions of logarithms of all nonzero values of quantitative variables
log_cols = quants.data.columns.drop(['YearBuilt', 'YearRemodAdd'])
plot_log_cont_dists(nrows=5, ncols=4, data=quants.data, log_cols=log_cols, figsize=(15, 20))
```


![png]({{site.baseurl}}/assets/images/explore_86_0.png)


### Relationships among quantitative variables


```python
# scatterplots of quantitative variables
sns.pairplot(quants.data)
```




    <seaborn.axisgrid.PairGrid at 0x121036da0>




![png]({{site.baseurl}}/assets/images/explore_88_1.png)


While pairplots can be helpful, this one is a bit too big to be of much use, although it may inform later methods of detecting relationships.

Some things do stand out:

- There appear to be regions of exclusion for certain pairs of variables, probably due to impossible values. For example, `YrRemodAdd` is never greater than `YrBuilt`. 

- Many of the distributions are very concentrated. `LotArea`, `BsmtFinSF2`, `LowQualFinSF`, `EnclosedPorch`, `3SsnPorch` all stand out as extremely concentrated.


Now we'll look at dependencies among the quantitative variables


```python
# dataframe of dependence distances of quantitative variables
quants_D_dep_df = D_dep(quants.data)
```


```python
# plot dependence distance
plot_D_dep(D_dep_df=quants_D_dep_df, figsize=(15, 10))
```


![png]({{site.baseurl}}/assets/images/explore_92_0.png)



```python
# plot lower dependence distances of quantitative variables
plot_low_D_dep(D_dep_df=quants_D_dep_df, D_threshold=0.8, figsize=(13, 8))
```


![png]({{site.baseurl}}/assets/images/explore_93_0.png)



```python
# display pairs of quantitatives with low dependence distance
rank_pairs_by_D(D_dep_df=quants_D_dep_df, D_threshold=0.8).head(10)
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
      <th>var1</th>
      <th>var2</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1stFlrSF</td>
      <td>GrLivArea</td>
      <td>0.158882</td>
    </tr>
    <tr>
      <td>2</td>
      <td>TotalBsmtSF</td>
      <td>1stFlrSF</td>
      <td>0.213738</td>
    </tr>
    <tr>
      <td>3</td>
      <td>LotArea</td>
      <td>GrLivArea</td>
      <td>0.227010</td>
    </tr>
    <tr>
      <td>4</td>
      <td>TotalBsmtSF</td>
      <td>GrLivArea</td>
      <td>0.242833</td>
    </tr>
    <tr>
      <td>5</td>
      <td>LotArea</td>
      <td>1stFlrSF</td>
      <td>0.250292</td>
    </tr>
    <tr>
      <td>6</td>
      <td>LotArea</td>
      <td>TotalBsmtSF</td>
      <td>0.269834</td>
    </tr>
    <tr>
      <td>7</td>
      <td>LotArea</td>
      <td>BsmtUnfSF</td>
      <td>0.273087</td>
    </tr>
    <tr>
      <td>8</td>
      <td>BsmtUnfSF</td>
      <td>TotalBsmtSF</td>
      <td>0.292260</td>
    </tr>
    <tr>
      <td>9</td>
      <td>BsmtUnfSF</td>
      <td>GrLivArea</td>
      <td>0.307987</td>
    </tr>
    <tr>
      <td>10</td>
      <td>BsmtUnfSF</td>
      <td>1stFlrSF</td>
      <td>0.320845</td>
    </tr>
  </tbody>
</table>
</div>


{% katexmm %}

Compared to quantitative and ordinal variables pairs, pairs of quantitative variables are showing much higher dependencies (lower dependence distances) overall. For many of these pairs , the high dependence makes sense given both variables are measuring very similar areas, for example, `1stFlrSF`, `GrLivArea` and `TotalBsmtSF`.

We expect that Pearsons' $\rho$ (i.e. correlation/linear dependence) of these variables should be high as well.

{% endkatexmm %}
```python
# plot pearson's correlation for quantitative variables
plot_corr(quants_data=quants.data, figsize=(15, 10))
```


![png]({{site.baseurl}}/assets/images/explore_97_0.png)



```python
# plot high absolute value of correlations of quantiatives
plot_high_corr(quants_data=quants.data, abs_corr_threshold=0.5, figsize=(15, 10))
```


![png]({{site.baseurl}}/assets/images/explore_98_0.png)



```python
# rank pairs of quantitatives by absolute values of correlation
rank_pairs_by_abs_corr_df = rank_pairs_by_abs_corr(quants_data=quants.data, abs_corr_threshold=0.5)
rank_pairs_by_abs_corr_df
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
      <th>var1</th>
      <th>var2</th>
      <th>abs_corr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>BsmtFinSF1</td>
      <td>TotalBsmtSF</td>
      <td>0.511258</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1stFlrSF</td>
      <td>GrLivArea</td>
      <td>0.546383</td>
    </tr>
    <tr>
      <td>3</td>
      <td>YearBuilt</td>
      <td>YearRemodAdd</td>
      <td>0.612023</td>
    </tr>
    <tr>
      <td>4</td>
      <td>YearRemodAdd</td>
      <td>GarageYrBlt</td>
      <td>0.618881</td>
    </tr>
    <tr>
      <td>5</td>
      <td>GarageYrBlt</td>
      <td>GarageArea</td>
      <td>0.653440</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2ndFlrSF</td>
      <td>GrLivArea</td>
      <td>0.658420</td>
    </tr>
    <tr>
      <td>7</td>
      <td>TotalBsmtSF</td>
      <td>1stFlrSF</td>
      <td>0.793482</td>
    </tr>
    <tr>
      <td>8</td>
      <td>YearBuilt</td>
      <td>GarageYrBlt</td>
      <td>0.808100</td>
    </tr>
  </tbody>
</table>
</div>



### Relationships between quantitatives and `SalePrice`


```python
# add log_SalePrice to quantitatives dataframe
quants.data['log_SalePrice'] = np.log(clean.data['SalePrice'])
quants.data['log_SalePrice']
```




           Id  
    train  1       12.247694
           2       12.109011
           3       12.317167
           4       11.849398
           5       12.429216
                     ...    
    test   2915          NaN
           2916          NaN
           2917          NaN
           2918          NaN
           2919          NaN
    Name: log_SalePrice, Length: 2916, dtype: float64


```python
# plot joint distributions of quantitative variables and log of sale price
plot_joint_dists_with_response(nrows=6, ncols=4, quants_data=quants.data, response='log_SalePrice', figsize=(15, 20))
```


![png]({{site.baseurl}}/assets/images/explore_103_0.png)


The distribution of some of the variables appears to be problematic for `seaborn` to fit a joint kernel density estimate. We'll look at scatterplots instead


```python
# scatterplots of quantitative variables and log of sale price
plot_scatter_with_response(nrows=6, ncols=4, quants_data=quants.data, response='log_SalePrice', figsize=(15, 20))
```


![png]({{site.baseurl}}/assets/images/explore_105_0.png)


Now will look at scatterplots of log transformations of the quantitive variables vs. `log_SalePrice`


```python
# scatterplots of log of nonzero values of quantitative variables and log of sale price
plot_log_scatter_with_response(nrows=6, ncols=4, quants_data=quants.data, response='log_SalePrice', figsize=(15, 20))
```


![png]({{site.baseurl}}/assets/images/explore_107_0.png)



```python
# rank dependence distance of quantiatives with log_SalePrice
D_dep_response(data=quants.data, response='log_SalePrice').sort_values(by='D').T
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
      <th>LotArea</th>
      <th>GrLivArea</th>
      <th>1stFlrSF</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>GarageArea</th>
      <th>BsmtFinSF1</th>
      <th>YearBuilt</th>
      <th>GarageYrBlt</th>
      <th>LotFrontage</th>
      <th>...</th>
      <th>YearRemodAdd</th>
      <th>WoodDeckSF</th>
      <th>MasVnrArea</th>
      <th>EnclosedPorch</th>
      <th>BsmtFinSF2</th>
      <th>ScreenPorch</th>
      <th>MiscVal</th>
      <th>LowQualFinSF</th>
      <th>3SsnPorch</th>
      <th>PoolArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>D</td>
      <td>0.166598</td>
      <td>0.216601</td>
      <td>0.243486</td>
      <td>0.259179</td>
      <td>0.266558</td>
      <td>0.390699</td>
      <td>0.41101</td>
      <td>0.549621</td>
      <td>0.561122</td>
      <td>0.579528</td>
      <td>...</td>
      <td>0.627967</td>
      <td>0.632256</td>
      <td>0.647403</td>
      <td>0.854684</td>
      <td>0.875006</td>
      <td>0.913641</td>
      <td>0.968622</td>
      <td>0.985514</td>
      <td>0.98756</td>
      <td>0.995242</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 22 columns</p>
</div>

{% katexmm %}

Considering the scatterplots and taking into account the dependence distance $D$, we see that some quantitative variables appear likely to be less helpful in predicting `SalePrice`. Of these, `EnclosedPorch`, `BsmtFinSF2`, `ScreenPorch`, `MiscVal`, `LowQualFinSF`, `3SSnPorch`, and `PoolArea` stand out (all have $D \gt 0.8$)


{% endkatexmm %}
