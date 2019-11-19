import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

raw_df = pd.read_csv('IMDB-Movie-Data.csv')
raw_array = raw_df.values

# extract x (input) and y (output) from the raw data
# process x (convert text to quantifiable values, check for missing data, etc.)

#### Linear Regression ####
features = raw_df[['Rating', 'Votes', 'Metascore']]
revenue = raw_df[['Revenue (Millions)']]
for data in raw_df.to_numpy():
    print(data)
features = features.to_numpy()
revenue = revenue.to_numpy()
K = 10
kf = KFold(n_splits=K)
# print(features)

featuresMean = features.mean(axis=0)
featuresStd = features.std(axis=0)
# print(featuresStd)
normFeatures = (features - featuresMean) / featuresStd
# print(normFeatures)
squareDistance = 0.0
for train_index, test_index in kf.split(features):
    featureTraining, featureTest = normFeatures[train_index], normFeatures[test_index]
    revenueTraining, revenueTest = revenue[train_index], revenue[test_index]
    LRModel = LinearRegression().fit(featureTraining, revenueTraining)
    predictData = LRModel.predict(featureTest)
    currentDistance = np.sum((revenueTest - predictData)**2)
    print(currentDistance)
    squareDistance += currentDistance
print(squareDistance / K)


LRModel = LinearRegression().fit(features, revenue)




# feature selection
# https://scikit-learn.org/stable/modules/feature_selection.html

# normalization
# code from assignments?

# regression models

# random forest
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# k-folds cross-validator
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html