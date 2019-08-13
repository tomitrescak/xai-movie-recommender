import React from 'react';
import logo from './logo.svg';
import './App.css';

import { client, ApolloProvider } from './config/apollo';
import { TopMoviesContainer } from './modules/top_movies/top_movies_container';

const App = () => (
  <ApolloProvider client={client}>
    <div>
      <h2>Recommender</h2>
      <TopMoviesContainer />
    </div>
  </ApolloProvider>
);

export default App;
