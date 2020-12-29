# datacook

A JavaScript library for feature engineering on datasets, it helps you to cook trainable datus out
as its name, [datacook][].

## Getting started

Install [datacook][] via NPM:

```js
npm install @pipcook/datacook --save
```

And write your first example:

```js
import { Image } from '@pipcook/datacook';

const dog = Image.read('test/node/image/artifacts/dog.jpg');
const data = [ ...img.resize(100, 100).data ];  // the 100x100 data.
```

Build your own word2vec model:

```js
import { Text } from '@pipcook/datacook';
const text = 'The king is a man who rules over a nation, he always have a woman beside him called the\
 queen.\n she helps the king controls the affars of the nation.\n Perhaps, she acclaimed the position of a king\
 when the king her husband is deceased.'.split('\n');
const stopWords = [ 'a', 'in', 'when', 'the', 'of', 'is', 'who' ];
const word2vec = new Text.Word2Vec(text, 5, stopWords);

// train the model
await word2vec.train();
word2vec.similarity('king', 'man'); // < 1.0
word2vec.mostSimilar('king'); // returns words and its weights.
```

## Contributing

To contribute to [datacook][], start from forking the repository, then clone to your local machine:

```shell
$ git clone https://github.com/imgcook/datacook.git
$ cd datacook
```

Install dependencies:

```shell
$ npm install
```

Run tests for both Node.js and browser environment:

```shell
npm run test
```

To run specific tests:

```shell
$ npm run test:node     # Node.js
$ npm run test:browser  # Browser
```

To build the source code to the dist folder, run:

```shell
$ npm run build
```

## License

Apache 2.0

[datacook]: https://github.com/imgcook/datacook
