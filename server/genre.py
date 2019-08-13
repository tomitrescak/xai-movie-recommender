import pandas as pd
from movies import loadMovies
from ast import literal_eval


def readGenres():
    try:
        return pd.read_pickle('./genres.pickle')
    except:
        md = loadMovies()
        s = md.apply(lambda x: pd.Series(x['genres']),
                     axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'genre'
        gen_md = md.drop('genres', axis=1).join(s)

        gen_md.to_pickle('genres.pickle')
        return gen_md


gen_md = readGenres()


def recommend_genre(genre, percentile=0.95):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()
                       ]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) &
                   (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('float')

    qualified['wr'] = qualified.apply(lambda x:
                                      (x['vote_count']/(x['vote_count']+m) *
                                       x['vote_average']) + (m/(m+x['vote_count']) * C),
                                      axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    return qualified


print(recommend_genre('Romance').head(10))
