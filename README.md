
This project analyzes the [Ames housing data](http://jse.amstat.org/v19n3/decock.pdf) and predicts the final sale price of houses
in that dataset. This is a learning project, based on the Kaggle Knowledge competition
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) but the aim is to experience and document all steps in an end-to-end predictive modeling problem it great detail.

The original dataset is available [here](http://www.amstat.org/publications/jse/v19n3/decock/AmesHousing.xls). A version of the dataset is available [on Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) - this is the dataset we'll be working with.

## Overview

The project consists of three stages, processing, exploratory analysis, and predictive modeling. It has the following directories:

- `/notebooks` - Jupyter notebooks for processing, exploratory analysis, and predictive modeling
- `/codes` - Supplemental code for the notebooks
- `/data` - Datasets.
- `/training` - Model training artifacts for persistence purposes
- `/submissions` - Model predictions for submission to Kaggle competition (neccesary for evaluating performance predictive models since Kaggle has the test set) 

## Data

There are several related datasets in `/data`:

- `train.csv`, `test.csv` - Original Kaggle train and test sets
- `orig.csv` - Full dataset with some metadata (`dtypes` and `MultiIndex`) for convenience in loading to `pandas.DataFrame`
- `clean.csv` - Processed and cleaned version of `orig.csv`

Both full and cleaned datasets are created and discussed in the notebook `process.ipynb`. They can also be built by running the script `process.py`