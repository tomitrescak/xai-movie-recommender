import React from 'react';

async function createBreakdown(movieSet: any[], keys: string[]) {
  return new Promise(accept => {
    const dictionaries: any = {};
    for (let key of keys) {
      dictionaries[key] = {};
    }

    for (let row of movieSet) {
      for (let key of keys) {
        // console.log('processing key: ' + key);
        // console.log(row);
        for (let item of row[key]) {
          if (dictionaries[key][item] >= 0) {
            dictionaries[key][item]++;
          } else {
            dictionaries[key][item] = 1;
          }
        }
      }
    }

    for (let key of keys) {
      dictionaries[key] = Object.keys(dictionaries[key]).map(mapKey => ({
        key: mapKey,
        value: dictionaries[key][mapKey]
      }));
      dictionaries[key].sort((a: any, b: any) => (a.value < b.value ? -1 : 1));
    }

    accept(dictionaries);
  });
}

type BreakdownProps = {
  movies: any[];
};
export const Breakdown: React.FC<BreakdownProps> = props => {
  const [processed, setProcessed] = React.useState<any>(null);

  if (!processed) {
    createBreakdown(props.movies, [
      'genres',
      'director',
      // 'production_companies',
      'production_countries',
      'spoken_languages',
      'screenplay',
      'producer',
      'e_producer',
      'cast_top_5'
    ]).then(breakdown => setProcessed(breakdown));

    return <div>Processing Breakdown ...</div>;
  }

  return <pre>{JSON.stringify(processed, null, 2)}</pre>;
};
