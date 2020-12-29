import '@tensorflow/tfjs-backend-cpu';

import Counter from './text/counter';
import CountVectorizer from './text/countvectorizer';
import Word2Vec from './text/word2vec';

export * as Generic from './generic';
export * as Image from './image/image-proc';
export const Text = {
  Counter,
  CountVectorizer,
  Word2Vec,
};
