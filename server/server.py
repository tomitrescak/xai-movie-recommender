from ariadne import QueryType, graphql_sync, make_executable_schema
from ariadne.constants import PLAYGROUND_HTML
from flask import Flask, request, jsonify
from flask_cors import CORS


from movie_recommender import top, getLatestData, similarTitles, titlesByUser, rate, userRatings


type_defs = """
    type Hello {
      message: String!
      other: String!
    }
    type Movie {
        adult: String
        belongs_to_collection: String
        budget: String
        cast_top_5: [String]
        director: [String]
        e_producer: [String]
        genres: [String]
        homepage: String
        id: String
        imdb_id: String
        keywords: [String]
        original_language: String
        original_title: String
        overview: String
        poster_path: String
        producer: [String]
        production_countries: [String]
        release_date: String
        revenue: Int
        runtime: Int
        screenplay: [String]
        spoken_languages: [String]
        tagline: String
        title: String
        vote_average: Float
        vote_count: Int
        wr: Float
        year: Int
        soup: String
    }
    type BreakDownItem {
        k: String
        v: String
    }
    type BreakDown {
        genres: [BreakDownItem]
        director: [BreakDownItem]
        production_companies: [BreakDownItem]
        production_countries: [BreakDownItem]
        spoken_languages: [BreakDownItem]
        screenplay: [BreakDownItem]
        producer: [BreakDownItem]
        e_producer: [BreakDownItem]
        cast_top_5: [BreakDownItem]
    }
    type Rating {
        id: Int
        rating: Float
        title: String
        poster_path: String
    }
    type ProcessingInfo {
        version: Int
        processing: Boolean
    }
    type Query {
        hello: Hello
        top(number: Int, quantile: Float): [Movie]
        similarTitles(title: String, number: Int, quantile: Float): [Movie]
        userTitles(userId: Int, title: String, number: Int): [Movie]
        userRatings(email: String): [Rating]
        rate(email: String, movieId: Int, rating: Int): Boolean
        latestData: ProcessingInfo
    }
"""

query = QueryType()


@query.field("hello")
def resolve_hello(_, info):
    request = info.context
    user_agent = request.headers.get("User-Agent", "Guest")
    return {
        "message": "Hello, %s!" % user_agent,
        "other": "yess"
    }

users = {
    "tomi": 200
}
id = 0

def findUser(email):
    global users, id
    if (email in users):
        return users[email]

    id = id + 1
    userId = 1000000 + id
    users[email] = userId

    return userId

@query.field("userRatings")
def resolve_userRatings(*_, email):
    userId = findUser(email)
    ratings = userRatings(userId)

    ratings = ratings.T.to_dict()
    ratings = ratings.values()

    return ratings;

@query.field("latestData")
def resolve_latestData(*_):
    return getLatestData()

@query.field("rate")
def resolve_userRatings(*_, email, movieId, rating):
    userId = findUser(email)
    rate(userId, movieId, rating)
    return True


@query.field("top")
def resolve_top(*_, number=20, quantile=0.9):
    # request = info.context
    pred = top(number, quantile)
    pred['idx'] = pred['id']
    di = pred.set_index('idx').T.to_dict()
    di = di.values()

    return di


@query.field("similarTitles")
def resolve_similar_titles(*_, title=None, number=20, quantile=0.9):
    # request = info.context
    pred = similarTitles(title, number, quantile)
    pred['idx'] = pred['id']
    di = pred.set_index('idx').T.to_dict()
    di = di.values()

    return di


@query.field("userTitles")
def resolve_user_titles(*_, userId=None, title=None, number=20):
    # request = info.context
    pred = titlesByUser(userId, title, number)
    pred['idx'] = pred['id']
    di = pred.set_index('idx').T.to_dict()
    di = di.values()

    return di


schema = make_executable_schema(type_defs, query)
app = Flask(__name__)
CORS(app)


@app.route("/graphql", methods=["GET"])
def graphql_playgroud():
    # On GET request serve GraphQL Playground
    # You don't need to provide Playground if you don't want to
    # but keep on mind this will not prohibit clients from
    # exploring your API using desktop GraphQL Playground app.
    return PLAYGROUND_HTML, 200


@app.route("/graphql", methods=["POST"])
def graphql_server():
    # GraphQL queries are always sent as POST
    data = request.get_json()

    # Note: Passing the request to the context is optional.
    # In Flask, the current request is always accessible as flask.request
    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
        debug=app.debug
    )

    status_code = 200 if success else 400
    json = jsonify(result)
    return json, status_code


if __name__ == "__main__":
    app.run(debug=True)
