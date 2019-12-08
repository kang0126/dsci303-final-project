"""
This file prepares the data and can be used for experiments using different models/parameters from util.py.

To run our algorithms:
    1. modify the code if you wish to change parameters (or leave it unchanged)
    2. run
       python3 experiments.py

Authors: Camille Kang, Isabella Huang, Jian Huang

Compatibility: python3 (Developed and tested using 3.7 only) | NOT compatible with python2
"""
from util import *

##### PREPARE DATA #####
tmdb = pd.read_csv('./data/tmdb.csv')

tmdb = feature_pre_process(tmdb)
clf_revenue = get_boolean_revenue(tmdb)

n_train = 1500

# below are the different combinations of features we could use
features = np.array(tmdb[['belongs_to_collection', 'budget', 'popularity', 'companyValue', 'runtime', 'starValue']])

features_with_processed_genres = np.array(tmdb[['belongs_to_collection', 'budget', 'popularity', 'companyValue',
                                                'runtime', 'starValue', 'popular_genre', 'unpopular_genre']])

# TODO modify this variable to change the features to use
features_to_use = features_with_processed_genres

# generate training/testing data without shuffling
# X_train, y_train = features_to_use[:n_train, :], clf_revenue[:n_train]
# X_test, y_test = features_to_use[n_train:, :], clf_revenue[n_train:]

# shuffling the data & generate training/testing data
perm = np.random.permutation(features_to_use.shape[0])
X_train, y_train = features_to_use[perm[:n_train], :], clf_revenue[perm[:n_train]]
X_test, y_test = features_to_use[perm[n_train:], :], clf_revenue[perm[n_train:]]


if __name__ == "__main__":
    exp_naive_bayes(X_train, y_train, X_test, y_test)
    exp_random_forest(X_train, y_train, X_test, y_test, n_trees=16)
