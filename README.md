
A learning project, based on the Kaggle knowledge competition
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The aim is to experience and document all the steps in an end-to-end predictive modeling problem in great detail.

Think of it as an overblown kernel :)

The original dataset is available [here](http://www.amstat.org/publications/jse/v19n3/decock/AmesHousing.xls). A version of the dataset is available [on Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) - this is the dataset we'll be working with.

## Overview

The project consists of three stages, processing, exploratory analysis, and predictive modeling. It has the following directories:

- `/notebooks` - Jupyter notebooks for processing, exploratory analysis, and predictive modeling
- `/codes` - Supplemental code for the notebooks
- `/data` - Datasets.
- `/training` - Model training artifacts for persistence purposes
- `/submissions` - Model predictions for submission to Kaggle competition (neccesary for evaluating performance predictive models since Kaggle has the test set) 

## Data

There are several related data files in `/data`:

- `train.csv`, `test.csv` - Original Kaggle train and test data
- `orig.csv` - Train and test data together with some metadata (`dtypes` and `MultiIndex`) for convenience in loading to `pandas.DataFrame`
- `clean.csv` - Processed and cleaned version of `orig.csv`

Both `orig.csv` and `clean.csv` are created and discussed in the notebook `process.ipynb`. They can also be built by running the script `process.py`