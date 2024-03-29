# DataCook

<img style="width: 350px; max-width: 100%" src="https://img.alicdn.com/imgextra/i2/O1CN01YJeTZV1dAoavsEwuP_!!6000000003696-2-tps-1069-353.png"/>

Machine learning and data science library for Javascript / Typescript.

---

## Getting started

### Dependencies

DataCook is built for javascript environment and can run in both [node.js](https://nodejs.org/) platform and browser. DataCook relies on [tensorflow.js](https://www.tensorflow.org/js) for basic numeric computation.

### Quick installation

DataCook can be installed by npm:

```bash
npm install @pipcook/datacook
```

or by yarn

```javascript
yarn add @pipcook/datacook
```

### Quick start: Train a simple linear-regression model

```javascript
import { Model } from '@pipcook/datacook';

const { LinearRegression } = Model;

const X = [
  [4, 5],
  [2, 3],
  [1, 4],
  [3, 8],
];
const y = [10, 5.5, 6.5, 12];
// create model
const lm = new LinearRegression();
// train linear model using feature set X and label set y
await lm.fit(X, y);
// get prediction
const yPred = lm.predict(X);
yPred.print();
// [10, 6, 6, 12]
```
