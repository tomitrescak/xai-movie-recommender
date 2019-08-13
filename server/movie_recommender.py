# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
import pickle

from movies import loadSmd, loadIndices, loadCosine, loadMovies, loadIndicesMap, loadRatings

import warnings
warnings.simplefilter('ignore')

print('Load smd')
smd = loadSmd()
print('Load indices')
indices = loadIndices(smd)
print('Load cosine')
cosine_sim = loadCosine(smd)
print('Load Indices map')
indices_map = loadIndicesMap(smd)
print('Load Ratings')
ratings = loadRatings()

#%%


def create_weighted_rating(m, C):
    def wr(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)
    return wr


def top(n, quantile=0.8):
    md = loadMovies()

    # this is V
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    # this is R
    vote_averages = md[md['vote_average'].notnull()
                       ]['vote_average'].astype('int')
    # this is C
    C = vote_averages.mean()
    m = vote_counts.quantile(quantile)
    weighted_rating = create_weighted_rating(m, C)

    qualified = md[(md['vote_count'] >= m) &
                   (md['vote_count'].notnull()) &
                   (md['vote_average'].notnull())]

    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('float')

    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    return qualified.head(n)


# TEST
# print(top(15, 0.95))

def similarTitles(title, number, quantile=0.8):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:number]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices]
    vote_counts = movies[movies['vote_count'].notnull()
                         ]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull(
    )]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(quantile)
    weighted_rating = create_weighted_rating(m, C)

    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) &
                       (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(number)
    return qualified


#%% TEST
# print(similarTitles('The Dark Knight', 500))
# print(improved_recommendations('Pulp Fiction'))

svd = None


def loadSavedSvd(id):
    try:
        return pickle.load(open('./cache/svd.pickle', 'rb'))
    except:
        return None


def initSvd():
    global svd
    if (svd != None):
        return svd
    svd = loadSavedSvd(0)
    if (svd != None):
        return svd
    #%%
    # surprise reader API to read the dataset
    reader = Reader()
    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']], reader)
    data.split(n_folds=5)
    svd = SVD()
    evaluate(svd, data, measures=['RMSE', 'MAE'])

    trainset = data.build_full_trainset()
    svd.train(trainset)

    pickle.dump(svd, open('./cache/svd.pickle', 'wb'))

    return svd


def predict(svd, userId):
    def fn(x):
        try:
            # print('creating map')
            # m = indices_map.loc[x]['movieId']
            # print('Predicting for ' + str(x) + ' and ' + str(m))
            res = svd.predict(userId, indices_map.loc[x]['movieId']).est
            # print('Predicted: ' + str(res))
            return res
        except:
            print('Error')
            return -1
    return fn


def titlesByUser(userId, title, number):
    svd = initSvd()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:number]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices]

    prediction = predict(svd, userId)
    movies['est'] = movies['id'].apply(prediction)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(number)


#%%
# print(titlesByUser(5000, 'Avatar', 500).head(10)['soup'])
#%%
# print(hybrid(5000, 'Avatar')['Titles'])


# #%%
