import React from 'react';
import ApolloClient from 'apollo-boost';

import { MockedProvider } from '@apollo/react-testing';

import { TOP_MOVIE_QUERY } from '../modules/top_movies/top_movies_container';
import top_movie_result from '../test/top.json';

export const client = new ApolloClient({
  uri: 'http://127.0.0.1:5000/graphql'
});
// export { ApolloProvider } from 'react-apollo';

const mocks = [
  {
    request: {
      query: TOP_MOVIE_QUERY,
      variables: { number: 500, quantile: 0.9 }
    },
    result: {
      data: {
        top: top_movie_result
      }
    }
  }
];

type MockProps = {
  client: any;
};

export const ApolloProvider: React.FC<MockProps> = ({ children }) => (
  <MockedProvider mocks={mocks} addTypename={false}>
    <>{children}</>
  </MockedProvider>
);
