import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import math



# Load the data and reference info data
raw_df = pd.read_csv('IMDB-Movie-Data.csv')
directors_df = pd.read_csv('Top-Directors.csv', encoding = "ISO-8859-1")
acts_df = pd.read_csv('Top-1000-Actors-Actresses.csv', encoding = "ISO-8859-1")
directors = directors_df['Name'].tolist()
acts = acts_df['Name'].tolist()
#
# tmdb = pd.read_csv('train.csv')
# print (tmdb)

# Create a new column "StarValue" in dataframe based on the director and actors/actresses
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
features = valid_df[['StarValue', 'Runtime (Minutes)','Rating', 'Votes', 'Metascore']]
revenue = valid_df[['Revenue (Millions)']]
# for data in raw_df.to_numpy():
#     print(data)
features = features.to_numpy()
revenue = revenue.to_numpy()


###############################################################################################
#                                   Feature Selection                                         #
###############################################################################################
print (np.median(revenue))


###############################################################################################
#                               Linear Regression with KFolds                                 #
###############################################################################################
# K = 10
# kf = KFold(n_splits=K)
# # print(features)
#
# featuresMean = features.mean(axis=0)
# featuresStd = features.std(axis=0)
#
# # print(featuresStd)
# normFeatures = (features - featuresMean) / featuresStd
# # print(normFeatures)
# squareDistance = 0.0
# for train_index, test_index in kf.split(features):
#     featureTraining, featureTest = normFeatures[train_index], normFeatures[test_index]
#     revenueTraining, revenueTest = revenue[train_index], revenue[test_index]
#     LRModel = LinearRegression().fit(featureTraining, revenueTraining)
#     predictData = LRModel.predict(featureTest)
#     currentDistance = np.sum((revenueTest - predictData)**2)
#     # print (np.concatenate((revenueTest, predictData), axis=1))
#     print (currentDistance)
#     squareDistance += currentDistance
# print(squareDistance / K)
#
#
# LRModel = LinearRegression().fit(features, revenue)




# feature selection
# https://scikit-learn.org/stable/modules/feature_selection.html

# normalization
# code from assignments?

# regression models
###############################################################################################
#                                       Naive Bayes                                           #
###############################################################################################

def applyNaiveBayes(X_train, y_train, X_test):
    # Feature Quantization
    training_median = np.median(X_train, axis=0)
    Q_train = X_train - training_median
    Q_train[Q_train < 0] = 0
    Q_train[Q_train > 0] = 1
    Q_test = X_test - training_median
    Q_test[Q_test < 0] = 0
    Q_test[Q_test > 0] = 1

    zeros = np.argwhere(y_train == 0)
    ones = np.argwhere(y_train == 1)

    n_train = 700
    num_zeros = len(zeros)
    num_ones = len(ones)
    Q_train_no = Q_train[zeros, :]
    Q_train_yes = Q_train[ones, :]

    # P(Feature Value = 0| y = 0)
    no_no = np.count_nonzero(Q_train_no == 0, axis=0) / num_zeros
    # P(Feature Value = 0| y = 1)
    no_yes = np.count_nonzero(Q_train_yes == 0, axis=0) / num_ones

    def predict(arr):
        no = num_zeros / n_train * np.prod(np.abs(arr - no_no))
        yes = num_ones / n_train * np.prod(np.abs(arr - no_yes))
        return int(no < yes)
    y_predict = np.apply_along_axis(predict, 1, Q_test)

    return y_predict


n_train = 700


def classify(num):
    if num < 50:
        return 0
    else:
        return 1
clf_revenue = np.apply_along_axis(classify, 1, revenue)

perm = np.random.permutation(features.shape[0])
X_train, y_train = features[perm[:n_train], :], clf_revenue[perm[:n_train]]
X_test, y_test = features[perm[n_train:], :], clf_revenue[perm[n_train:]]
# X_train, y_train = features[:n_train, :], clf_revenue[:n_train]
# X_test, y_test = features[n_train:, :], clf_revenue[n_train:]
y_hat = applyNaiveBayes(X_train, y_train, X_test)
print('Naive bayes test error: %g' % (y_test != y_hat).mean())

###############################################################################################
#                                       Random Forest                                         #
###############################################################################################
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
n_train = 700

def classify(num):
    if num < 50: return 0
    else: return 1
    
clf_revenue = np.apply_along_axis(classify, 1, revenue)

X_train, y_train = features[:n_train, :], clf_revenue[:n_train]
X_test, y_test = features[n_train:, :], clf_revenue[n_train:]


"""
# num_trees_range = range(2,40,2)
# oob_score_record = []
# for num_trees in num_trees_range:
#     clf = RandomForestClassifier(n_estimators = num_trees, oob_score= True)
#     # clf.fit(X_train, y_train)
#     clf.fit(features, clf_revenue)
#
#     oob_score_record.append(clf.oob_score_)
# plt.scatter(num_trees_range, oob_score_record)
# plt.show()
#
# idx = np.argmax(oob_score_record)
# num_trees = num_trees_range[idx]
# print ("Using {} trees to train the random forest model has oob score {}".format(num_trees, oob_score_record[idx]))
"""


clf = RandomForestClassifier(n_estimators = 32, oob_score= True)
clf.fit(X_train,y_train)
# clf.fit(features, clf_revenue)
y_predict = clf.predict(X_test)
print ('Random forest test error: %g' % (y_predict != y_test).mean())
Pclass = (clf_revenue[:, None] == np.arange(2)[None, :]).mean(0)
print ('Majority class error: %g' % (clf_revenue != np.argmax(Pclass)).mean())

"""



# k-folds cross-validator
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

