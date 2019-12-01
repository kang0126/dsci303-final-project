from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

tmdb = pd.read_csv('tmdb.csv')
tmdb = tmdb[tmdb['budget'] > 100]
tmdb = tmdb[tmdb['original_language'] == 'en']

directors_df = pd.read_csv('Top-Directors.csv', encoding="ISO-8859-1")
acts_df = pd.read_csv('Top-1000-Actors-Actresses.csv', encoding="ISO-8859-1")
directors = directors_df['Name'].tolist()
acts = acts_df['Name'].tolist()
companies = ['paramount', 'mgm', 'disney', 'dreamworks', 'twentieth', 'universal', 'lionsgate', 'warner', 'columbia']
genres_sorted = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Crime', 'Adventure', 'Science Fiction', 'Horror',
                 'Family', 'Mystery', 'Fantasy', 'Animation', 'History', 'Music', 'War', 'Western', 'Documentary',
                 'Foreign', 'TV Movie']
genres_counter = Counter(
    {'Drama': 935, 'Comedy': 653, 'Thriller': 581, 'Action': 534, 'Romance': 351, 'Crime': 348, 'Adventure': 345,
     'Science Fiction': 232, 'Horror': 227, 'Family': 187, 'Mystery': 167, 'Fantasy': 167, 'Animation': 85,
     'History': 80, 'Music': 67, 'War': 64, 'Western': 29, 'Documentary': 19, 'Foreign': 4, 'TV Movie': 1})

# features = tmdb[['belongs_to_collection', 'budget', 'genres', 'original_language', 'popularity', 'production_companies',
#                  'runtime', 'cast']]
# revenue = tmdb['revenue']

tmdb['belongs_to_collection'] = tmdb['belongs_to_collection'].apply(lambda x: int(type(x) is not float))


# tmdb['original_language'] = tmdb['original_language'].apply(lambda x: x == 'en')


def starV(str):
    if type(str) is float:
        return 0

    str_list = list(filter(lambda x: x.startswith("'name':"), str.split(", ")))
    val = 0
    for part in str_list:
        star = part[9: -1]
        if star in acts or star in directors:
            val += 1
    return val


def companyV(str):
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


def str2list(str):
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


tmdb['starValue'] = tmdb['cast'].apply(starV)
tmdb['companyValue'] = tmdb['production_companies'].apply(companyV)
tmdb['genres_list'] = tmdb['genres'].apply(str2list)


# print (tmdb['genres_list'])

def convert_genre(genre_lst, genre):
    return 1 if genre in genre_lst else 0


for genre in genres_sorted:
    tmdb[genre] = tmdb['genres_list'].apply(lambda x: 1 if genre in x else 0)

features = np.array(tmdb[['belongs_to_collection', 'budget', 'popularity', 'companyValue', 'runtime', 'starValue']])
revenue = np.array(tmdb[['revenue']])

features_with_binary_genres = np.array(
    tmdb[['belongs_to_collection', 'budget', 'popularity', 'companyValue', 'runtime', 'starValue'] + genres_sorted])

genre_box_plot_data = []
genres_to_plot = genres_sorted[0:5] + genres_sorted[6:15]
for genre in genres_to_plot:
    movies = np.array(tmdb[tmdb[genre] == 1]['revenue'])
    genre_box_plot_data.append(movies)
    # print('=-=-=-=-=-=-=-=-=-=')
    # print(genre)
    # print(np.mean(movies))
    # print(np.median(movies))
    # print(np.std(movies))

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Save the figure
# fig.savefig('fig1.png', bbox_inches='tight')

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

ax.set_xticklabels(genres_to_plot)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.show()


def classify(num):
    if num < 3.5e7:
        return 0
    else:
        return 1


clf_revenue = np.apply_along_axis(classify, 1, revenue)

n_train = 1500
perm = np.random.permutation(features.shape[0])
X_train, y_train = features[perm[:n_train], :], clf_revenue[perm[:n_train]]
X_test, y_test = features[perm[n_train:], :], clf_revenue[perm[n_train:]]


# perm = np.random.permutation(features_with_binary_genres.shape[0])
# X_train, y_train = features_with_binary_genres[perm[:n_train], :], clf_revenue[perm[:n_train]]
# X_test, y_test = features_with_binary_genres[perm[n_train:], :], clf_revenue[perm[n_train:]]

# X_train, y_train = features[:n_train, :], clf_revenue[:n_train]
# X_test, y_test = features[n_train:, :], clf_revenue[n_train:]


def applyNaiveBayes(X_train, y_train, X_test):
    # quantized_train_features = X_train[:, [0,3]]
    # quantized_test_features = X_test[:, [0,3]]
    #
    # X_train = X_train[:,[1,2,4,5]]
    # X_test = X_test[:, [1, 2, 4, 5]]

    # Feature Quantization
    training_median = np.median(X_train, axis=0)
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

    y_predict = np.apply_along_axis(predict, 1, Q_test)

    return y_predict


y_hat = applyNaiveBayes(X_train, y_train, X_test)
print('Naive bayes test error: %g' % (y_test != y_hat).mean())


def applyRandomForest(X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators=20, oob_score=True)
    clf.fit(X_train, y_train)
    # clf.fit(features, clf_revenue)
    y_predict = clf.predict(X_test)
    print(clf.feature_importances_)
    return y_predict


y_hat = applyRandomForest(X_train, y_train, X_test)
print ('Random forest test error: %g' % (y_hat != y_test).mean())
