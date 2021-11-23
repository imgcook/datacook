import Counter from './text/counter';
import CountVectorizer from './text/countvectorizer';
import Word2Vec from './text/word2vec';
import { LabelEncoder, OneHotEncoder } from './tabular/encoder';

export * as Dataset from './dataset';
export * as Generic from './generic';
export { Image } from './image';
export * as Rand from './rand';
export * as Model from './model';
export * as Metrics from './metrics';
export * as Linalg from './linalg';
export * as Preprocess from './preprocess';
export * as Stat from './stat';

export const Text = {
  Counter,
  CountVectorizer,
  Word2Vec
};

export const Encoder = {
  LabelEncoder,
  OneHotEncoder
};

