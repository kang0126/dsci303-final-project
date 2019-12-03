from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

tmdb = pd.read_csv('tmdb.csv')

###### FEATURE PRE-PROCESSING #######

# BUDGET - remove data entries with unreasonable budgets
tmdb = tmdb[tmdb['budget'] > 100]

# LANGUAGE - remove data entries that are non-English
tmdb = tmdb[tmdb['original_language'] == 'en']

# BELONGS TO A COLLECTION - boolean value indicating whether a movie belongs to a collection
tmdb['belongs_to_collection'] = tmdb['belongs_to_collection'].apply(lambda x: int(type(x) is not float))

# DIRECTORS - *** NOT USED ***
directors_df = pd.read_csv('Top-Directors.csv', encoding="ISO-8859-1")
directors = directors_df['Name'].tolist()

# ACTORS - STAR VALUES
acts_df = pd.read_csv('Top-1000-Actors-Actresses.csv', encoding="ISO-8859-1")
acts = acts_df['Name'].tolist()


def starV(str):
    """
    returns a star value based on TODO...
    """
    if type(str) is float:
        return 0

    str_list = list(filter(lambda x: x.startswith("'name':"), str.split(", ")))
    val = 0
    for part in str_list:
        star = part[9: -1]
        if star in acts or star in directors:
            val += 1
    return val


tmdb['starValue'] = tmdb['cast'].apply(starV)

# PRODUCTION COMPANIES - major companies
companies = ['paramount', 'mgm', 'disney', 'dreamworks', 'twentieth', 'universal', 'lionsgate', 'warner', 'columbia']


def companyV(str):
    """
    returns boolean value indicating whether a movie was produced by one of the major companies.
    """
    if type(str) is float:
        return 0

    str_list = list(filter(lambda x: x.startswith("{'name':"), str[1:-1].split(", ")))
    for part in str_list:
        company = part[10:-1].lower()
        if any(keyword in company for keyword in companies):
            return 1
        else:
            continue
    return 0


tmdb['companyValue'] = tmdb['production_companies'].apply(companyV)

# GENRES - a) 20 boolean values for each genre  b) 2 boolean variables indicating popular & unpopular
genres_counter = Counter(
    {'Drama': 935, 'Comedy': 653, 'Thriller': 581, 'Action': 534, 'Romance': 351, 'Crime': 348, 'Adventure': 345,
     'Science Fiction': 232, 'Horror': 227, 'Family': 187, 'Mystery': 167, 'Fantasy': 167, 'Animation': 85,
     'History': 80, 'Music': 67, 'War': 64, 'Western': 29, 'Documentary': 19, 'Foreign': 4, 'TV Movie': 1})

genres_sorted = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Crime', 'Adventure', 'Science Fiction', 'Horror',
                 'Family', 'Mystery', 'Fantasy', 'Animation', 'History', 'Music', 'War', 'Western', 'Documentary',
                 'Foreign', 'TV Movie']

# genres that have a median of >= $35 million
genres_popular = ['Comedy', 'Thriller', 'Action', 'Crime', 'Adventure', 'Science Fiction', 'Family', 'Fantasy',
                  'Animation']

# genres that have a median of < $35 million
genres_unpopular = ['Drama', 'Romance', 'Horror', 'Mystery', 'History', 'Music', 'War', 'Western', 'Documentary',
                    'Foreign', 'TV Movie']


def str2list(str):
    """
    returns a list of genres for each movie
    """
    if type(str) is float:
        return []

    str_list = list(filter(lambda x: x.startswith("'name':"), str.split(", ")))
    genre_list = []

    for i in range(len(str_list)):
        part = str_list[i]
        if i == len(str_list) - 1:
            genre = part[9: -3]
        else:
            genre = part[9:-2]
        genre_list.append(genre)

    return genre_list


tmdb['genres_list'] = tmdb['genres'].apply(str2list)

for genre in genres_sorted:
    tmdb[genre] = tmdb['genres_list'].apply(lambda x: 1 if genre in x else 0)


def share_common_element(lst0, lst1):
    for a in lst0:
        if a in lst1:
            return True
    return False


tmdb['popular_genre'] = tmdb['genres_list'].apply(lambda x: 1 if share_common_element(x, genres_popular) else 0)
tmdb['unpopular_genre'] = tmdb['genres_list'].apply(lambda x: 1 if share_common_element(x, genres_unpopular) else 0)

# REVENUE - boolean value indicating if above median ($35 million)
revenue = np.array(tmdb[['revenue']])


def classify(num):
    if num < 3.5e7:
        return 0
    else:
        return 1


clf_revenue = np.apply_along_axis(classify, 1, revenue)

##### Training/Testing Data #####
n_train = 1500

# below are the different combinations of features we could use
features = np.array(tmdb[['belongs_to_collection', 'budget', 'popularity', 'companyValue', 'runtime', 'starValue']])

features_with_binary_genres = np.array(
    tmdb[['belongs_to_collection', 'budget', 'popularity', 'companyValue', 'runtime', 'starValue'] + genres_sorted])

features_with_processed_genres = np.array(tmdb[['belongs_to_collection', 'budget', 'popularity', 'companyValue',
                                                'runtime', 'starValue', 'popular_genre', 'unpopular_genre']])

# TODO modify this variable to change the features to use
features_to_use = features_with_processed_genres

# generate training/testing data
# X_train, y_train = features_to_use[:n_train, :], clf_revenue[:n_train]
# X_test, y_test = features_to_use[n_train:, :], clf_revenue[n_train:]

# shuffling the data & generate training/testing data
perm = np.random.permutation(features_to_use.shape[0])
X_train, y_train = features_to_use[perm[:n_train], :], clf_revenue[perm[:n_train]]
X_test, y_test = features_to_use[perm[n_train:], :], clf_revenue[perm[n_train:]]


##### MODELS/ALGORITHMS #####

# MODEL EVALUATION
def compute_f1(true, pred):
    total = len(true)
    tp = tn = fp = fn = 0
    for i in range(total):
        if true[i] == pred[i]:
            if true[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if true[i] == 1:
                fn += 1
            else:
                fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(tp, tn, fp, fn)

    return precision, recall, 2 * precision * recall / (precision + recall)


# NAIVE BAYES
def applyNaiveBayes(X_train, y_train, X_test):
    # quantized_train_features = X_train[:, [0,3]]
    # quantized_test_features = X_test[:, [0,3]]
    #
    # X_train = X_train[:,[1,2,4,5]]
    # X_test = X_test[:, [1, 2, 4, 5]]

    # Feature Quantization
    training_median = np.median(X_train, axis=0)

    # TODO MODIFY THIS IF FEATURES ARE CHANGED
    # cast the medians of boolean variables to 0.5
    for i in [0, 3, 6, 7]:
        training_median[i] = 0.5

    Q_train = X_train - training_median
    Q_train[Q_train < 0] = 0
    Q_train[Q_train > 0] = 1
    Q_test = X_test - training_median
    Q_test[Q_test < 0] = 0
    Q_test[Q_test > 0] = 1

    # Q_train = np.concatenate((Q_train, quantized_train_features), axis = 1)
    # Q_test = np.concatenate((Q_test, quantized_test_features), axis=1)

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

    y_predict_training = np.apply_along_axis(predict, 1, Q_train)
    y_predict = np.apply_along_axis(predict, 1, Q_test)

    return y_predict_training, y_predict


print('=-=-=-= Naive Bayes =-=-=-=')
y_hat_train, y_hat_test = applyNaiveBayes(X_train, y_train, X_test)

# training error
print('Training Error: %g' % np.mean(y_train != y_hat_train))
p, r, f1 = compute_f1(y_train, y_hat_train)
print('Precision: {}'.format(p))
print('Recall: {}'.format(r))
print('F1 Score: {}\n'.format(f1))

# testing error
print('Testing Error: %g' % np.mean(y_test != y_hat_test))
p, r, f1 = compute_f1(y_test, y_hat_test)
print('Precision: {}'.format(p))
print('Recall: {}'.format(r))
print('F1 Score: {}\n'.format(f1))


# RANDOM FOREST
def applyRandomForest(X_train, y_train, X_test, n_trees=16):
    clf = RandomForestClassifier(n_estimators=n_trees, oob_score=True)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    y_predict_training = clf.predict(X_train)
    return y_predict_training, y_predict, clf.oob_score_, clf.feature_importances_


print('=-=-=-= Random Forest =-=-=-=')
y_hat_training, y_hat_test, obb_score, feature_importances = applyRandomForest(X_train, y_train, X_test, 16)

# Training Error
print ('Training Error: %g' % np.mean((y_hat_training != y_train)))
p, r, f1 = compute_f1(y_train, y_hat_training)
print('Precision: {}'.format(p))
print('Recall: {}'.format(r))
print('F1 Score: {}\n'.format(f1))

# Testing Error
print ('Testing Error: %g' % np.mean((y_hat_test != y_test)))
p, r, f1 = compute_f1(y_test, y_hat_test)
print('Precision: {}'.format(p))
print('Recall: {}'.format(r))
print('F1 Score: {}\n'.format(f1))

print(feature_importances)


def choose_trees_plot():
    oob = []
    for n_trees in range(2, 22, 2):
        y_hat_train, y_hat, obb_score, feature_importances = applyRandomForest(X_train, y_train, X_test, n_trees)
        oob.append(obb_score)

    plt.plot(range(2, 22, 2), oob)
    plt.xlabel('Number of Trees')
    plt.ylabel('OOB Scores')
    plt.show()


# FEEDFORWARD NEURAL NETWORK
# nnclf = MLPClassifier(hidden_layer_sizes=(10,),
#                       activation='relu',
#                       solver='lbfgs',
#                       )
# nnclf.fit(X_train, y_train)
# print(nnclf.score(X_test, y_test))


##### VISUALIZATION #####
def plot_genre_box():
    genre_box_plot_data = []
    genres_to_plot = genres_sorted[0:5] + genres_sorted[6:15]
    for genre in genres_to_plot:
        movies = np.array(tmdb[tmdb[genre] == 1]['revenue']) / 1000000
        genre_box_plot_data.append(movies)
        # print('=-=-=-=-=-=-=-=-=-=')
        # print(genre)
        # print(np.mean(movies))
        # print(np.median(movies))
        # print(np.std(movies))

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)

    bp = ax.boxplot(genre_box_plot_data, patch_artist=True)
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    xlabel = genres_to_plot[:]
    xlabel[6] = 'SciFi'
    ax.set_xticklabels(xlabel)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.hlines(35, xmin=1, xmax=14, color='r')

    plt.show()

