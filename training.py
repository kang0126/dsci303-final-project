import numpy as np
import pandas as pd

raw_df = pd.read_csv('IMDB-Movie-Data.csv')
raw_array = raw_df.values

# extract x (input) and y (output) from the raw data
# process x (convert text to quantifiable values, check for missing data, etc.)

# feature selection
# https://scikit-learn.org/stable/modules/feature_selection.html

# normalization
# code from assignments?

# regression models

# random forest
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# k-folds cross-validator
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html