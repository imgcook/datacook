---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
permalink: /
title: Home
nav_order: 1
---
# DataCook

Machine learning and data science library for Javascript / Typescript.
{: .fs-6 .fw-300 }

[Get started now](#getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 } [View it on GitHub](https://github.com/imgcook/datacook){: .btn .fs-5 .mb-4 .mb-md-0 }

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
import {LinearRegression} from '@pipcook/datacook';

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
