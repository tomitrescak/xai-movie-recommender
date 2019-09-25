import pandas as pd
import numpy as np
from os.path import exists

from ast import literal_eval
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

import os
if not os.path.exists('cache'):
    os.makedirs('cache')


def loadCredits():
    credits = pd.read_csv('./the-movies-dataset/credits.csv')
    credits['id'] = credits['id'].astype('int')
    return credits


def loadKeywords():
    keywords = pd.read_csv('./the-movies-dataset/keywords.csv')
    keywords['id'] = keywords['id'].astype('int')
    return keywords


def loadLinks():
    links_small = pd.read_csv('./the-movies-dataset/links_small.csv')
    links_small = links_small[links_small['tmdbId'].notnull()
                              ]['tmdbId'].astype('int')

    return links_small


md = None


def loadMovies():
    global md
    if (md is not None):
        return md
    try:
        md = pd.read_pickle('./cache/movies.pickle')
        print('Movies de-pickled.')
    except:
        print('Loading movies ...')
        md = pd.read_csv('./the-movies-dataset/movies_metadata.csv')

        keywords = loadKeywords()
        credits = loadCredits()

        def convert_int(x):
            try:
                return int(x)
            except:
                return np.nan

        def extract_name(x):
            return [i['name'] for i in x] if isinstance(x, list) else []

        def get_job(job):
            def fn(m):
                return list(filter(lambda x: x['job'] == job, m))
            return fn

        def cc(x):
            if isinstance(x, dict):
                return x['name']
            else:
                return ''

        md['id'] = md['id'].apply(convert_int)
        md[md['id'].isnull()]
        md = md.drop([19730, 29503, 35587])
        md['id'] = md['id'].astype('int')

        md = md.merge(credits, on='id')
        md = md.merge(keywords, on='id')

        md['tagline'] = md['tagline'].fillna('')
        md['description'] = md['overview'] + md['tagline']
        md['description'] = md['description'].fillna('')

        md['cast'] = md['cast'].apply(literal_eval)
        md['crew'] = md['crew'].apply(literal_eval)
        md['keywords'] = md['keywords'].apply(literal_eval)
        md['cast_size'] = md['cast'].apply(lambda x: len(x))
        md['crew_size'] = md['crew'].apply(lambda x: len(x))

        md['cast'] = md['cast'].apply(extract_name)
        md['cast_top_5'] = md['cast'].apply(
            lambda x: x[:5] if len(x) >= 3 else x)
        md['keywords'] = md['keywords'].apply(extract_name)
        md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
            lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

        md['belongs_to_collection'] = md['belongs_to_collection'].fillna(
            '{ "name": ""}').apply(literal_eval).apply(cc)
        md['genres'] = md.fillna('[]')['genres'].apply(
            literal_eval).apply(extract_name)
        md['spoken_languages'] = md['spoken_languages'].fillna('[]').apply(
            literal_eval).apply(extract_name)

        md['director'] = md['crew'].apply(
            get_job('Director')).apply(extract_name)
        md['director'].head()

        md['screenplay'] = md['crew'].apply(
            get_job('Screenplay')).apply(extract_name)
        md['screenplay'].head()

        md['producer'] = md['crew'].apply(
            get_job('Producer')).apply(extract_name)
        md['producer'].head()

        md['e_producer'] = md['crew'].apply(
            get_job('Executive Producer')).apply(extract_name)
        md['e_producer'].head()

        md['production_companies'] = md['production_companies'].fillna('[]').apply(
            literal_eval).apply(extract_name)

        md['production_countries'] = md['production_countries'].fillna('[]').apply(
            literal_eval).apply(extract_name)

        md.to_pickle('./cache/movies.pickle')

    return md


def loadRatings():
    if exists('./the-movies-dataset/ratings_modified.csv'):
        ratings = pd.read_csv('./the-movies-dataset/ratings_modified.csv')
    else:
        ratings = pd.read_csv('./the-movies-dataset/ratings_small.csv')

    return ratings


smd = None


def loadSmd():
    global smd
    try:
        # if smd != None:
        #     print('Found in-memory smd')
        #     return smd
        smd = pd.read_pickle('./cache/smd.pickle')
        print('Smd de-pickled.')
        return smd
    except:
        print('Loading smd ...')
        md = loadMovies()
        links_small = loadLinks()

        def convert_int(x):
            try:
                return int(x)
            except:
                return np.nan

        smd = md[md['id'].isin(links_small)]

        smd['cast_soup'] = smd['cast_top_5'].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x])
        smd['director_soup'] = smd['director'].astype('str').apply(
            lambda x: str.lower(x.replace(" ", "")))
        smd['director_soup'] = smd['director_soup'].apply(lambda x: [x, x, x])

        s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack(
        ).reset_index(level=1, drop=True)
        s.name = 'keyword'
        s = s.value_counts()
        s = s[s > 1]
        # Just an example
        stemmer = SnowballStemmer('english')

        def filter_keywords(x):
            words = []
            for i in x:
                if i in s:
                    words.append(i)
            return words

        smd['keywords_soup'] = smd['keywords'].apply(filter_keywords)
        smd['keywords_soup'] = smd['keywords_soup'].apply(
            lambda x: [stemmer.stem(i) for i in x])
        smd['keywords_soup'] = smd['keywords_soup'].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x])
        smd['soup'] = smd['keywords_soup'] + smd['cast_soup'] + \
            smd['director_soup'] + smd['genres']
        smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

        smd.to_pickle('./cache/smd.pickle')
    return smd


def loadIndices(smd):
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])

    return indices


cosine_sim = None


def loadCosine(smd):
    global cosine_sim
    if (cosine_sim != None):
        return cosine_sim

    try:
        cosine_sim = pickle.load(open('./cache/cosine.pickle', 'rb'))
    except:
        count = CountVectorizer(analyzer='word', ngram_range=(
            1, 2), min_df=0, stop_words='english')
        count_matrix = count.fit_transform(smd['soup'])

        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        pickle.dump(cosine_sim, open('./cache/cosine.pickle', 'wb'))

    return cosine_sim


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


def loadIndicesMap(smd):
    id_map = pd.read_csv(
        './the-movies-dataset/links_small.csv')[['movieId', 'tmdbId']]
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
    # id_map = id_map.set_index('tmdbId')
    indices_map = id_map.set_index('id')

    return indices_map
