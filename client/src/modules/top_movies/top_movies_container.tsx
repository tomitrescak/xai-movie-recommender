import React from 'react';

import { useQuery } from '@apollo/react-hooks';
import { gql } from 'apollo-boost';
import { Breakdown } from './breakdown';

export const TOP_MOVIE_QUERY = gql`
  query TopMovies($number: Int, $quantile: Float) {
    top(number: $number, quantile: $quantile) {
      title
      adult
      belongs_to_collection
      budget
      cast_top_5
      director
      e_producer
      genres
      homepage
      id
      imdb_id
      keywords
      original_language
      original_title
      overview
      producer
      production_countries
      release_date
      revenue
      runtime
      screenplay
      spoken_languages
      tagline
      vote_average
      vote_count
      year
      wr
    }
  }
`;

export function TopMoviesContainer() {
  const { loading, error, data } = useQuery(TOP_MOVIE_QUERY, {
    variables: { number: 500, quantile: 0.9 }
  });

  if (loading) return <p>Loading...</p>;
  if (error)
    return (
      <p>
        Error :( <pre>{JSON.stringify(error)}</pre>
      </p>
    );

  // calculate breakdowns

  return (
    <div>
      <h3>Top Movies</h3>
      <Breakdown movies={data.top} />
    </div>
  );
}
