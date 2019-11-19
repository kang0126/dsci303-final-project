import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

raw_df = pd.read_csv('IMDB-Movie-Data.csv')
directors_df = pd.read_csv('Top-Directors.csv', encoding = "ISO-8859-1")
acts_df = pd.read_csv('Top-1000-Actors-Actresses.csv', encoding = "ISO-8859-1")

directors = directors_df['Name'].tolist()
acts = acts_df['Name'].tolist()

def starV(ppl):
    val = 0
    ppl_list = ppl.split(", ")
    for star in ppl_list:
        if star in directors or star in acts:
            val += 1
    return val

raw_df['StarValue'] = raw_df.apply(lambda x: starV(x['Director'])+ starV(x['Actors']), axis=1)

# remove lines with missing data
valid_df = raw_df.dropna()
valid_array = valid_df.values

# extract x (input) and y (output) from the raw data


#### Linear Regression ####
features = valid_df[['StarValue','Rating', 'Votes', 'Metascore']]
revenue = valid_df[['Revenue (Millions)']]
# for data in raw_df.to_numpy():
#     print(data)
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