---
layout: default
title: Linear Regression
parent: Models
lang: en
---

# Linear Regression

Linear regression is a classical **supervised learning** algorithm used when target / dependence variable is **continuous real number**. Linear regression fits a linear relationship between dependent variable $y$ and one or more independent variable $X$.

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 +... + \beta_m x_m +\epsilon
$$

where $\epsilon$ is the random residual term.

Linear regression model is fitted by minimizing the Mean squared error (MSE) between predicted value $\hat y$ and real target $y$:

$$
\min_\beta \sum_{i=1}^n ||\hat y_i - y_i||^2
$$

In this implementation of **LinearRegression**, we use stochastic gradient descent (SGD) to minimize MSE. SGD is quite adaptable for most of the cases, whenever your data or feature size is large or small, however it may not be efficient sometimes.

If you want to get statistical inference results for linear regression or more efficient fitting (especially for moderate data and feature size), please refer to [LinearRegressionAnalysis](../linear-regression-analysis) model, in which **Ordinary Least Square (OLS)** method is applied.

## Import

```javascript
import * as Datacook from 'datacook';
const { LinearRegression } = DataCook.Model;
```

## Constructor

```typescript
lm = new LinearRegression({ optimizerType: 'adam' });
```

### Option Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
|   fitIntercept `optional`   |  boolean  |     Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations        |
| normalize  `optional`| boolean | This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. |
| optimizerType  `optional`|  'sgd' \| 'momentum' \| 'adagrad' \| 'adadelta' \| 'adam' \| 'adamax' \| 'rmsprop' | optimizer types for training. All of the following [optimizers types](https://js.tensorflow.org/api/latest/#Training-Optimizers) supported in tensorflow.js can be applied. **Default to 'adam'** |
| optimizerProps  `optional`| OptimizerProps | parameters used to init corresponding optimizer, you can refer to [documentations in tensorflow.js](https://js.tensorflow.org/api/latest/#Training-Optimizers) to find the supported initailization paratemters for a given type of optimizer. For example, `{ learningRate: 0.1, beta1: 0.1, beta2: 0.2, epsilon: 0.1 }` could be used to initialize adam optimizer.|  


## Methods

### fit

Fit linear regression model according to X, y.

```typescript
async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>,
    params: LinearRegerssionTrainParams = { batchSize: 32, epochs: -1 })
```

#### Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData     | Tensor \| RecursiveArray<number> | input data of shape (nSamples,nFeatures) in type of array or tensor |
| yData     | Tensor \| RecursiveArray<number> | Tensor like of shape (n_sample, ), input target values |
| params `optional` | LinearRegerssionTrainParams | `batchSize`: batch size: **default = 32**, `maxIterTimes`: max iteration times, **default = 20000** |


#### Returns

[LinearRegression](#LinearRegression)

### trainOnBatch

Train your model on batch. If your dataset is large, it could be a better choice than [fit](#fit) directly. 


#### Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData     | Tensor \| RecursiveArray<number> | input data of shape (nSamples,nFeatures) in type of array or tensor |
| yData     | Tensor \| RecursiveArray<number> | Tensor like of shape (n_sample, ), input target values |


#### Returns

[LinearRegression](#LinearRegression)

### predict

```typescript
async predict(xData: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData     | Tensor \| RecursiveArray<number> | input data of shape (nSamples,nFeatures) in type of array or tensor |


#### Returns

Tensor of predicted values


### getCoef

Get linear regression coefficients

```typescript
getCoef(): { coefficients: Tensor, intercept: Tensor }
```



#### Returns

{ coefficients: Tensor, intercept: Tensor }

### initModelFromWeights

```typescript
initModelFromWeights(inputShape: number, 
    useBias: boolean, 
    weights: (Float32Array | Int32Array | Uint8Array)[]): void
```
#### Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| inputShape   | number | size of input features |
| useBias | boolean | if use bias |
| weights | number[][] | weights to initialize |

### fromJson

Load model paramters from json string object

```typescript
async fromJson(modelJson: string): Promise<LinearRegression>
```

#### Returns

LinearRegression

### toJson

Dump model parameters to json string.

```typescript
async toJson(): Promise<string>
```

#### Returns

Stringfied model parameters

## Examples

```javascript
import * as datacook from 'datacook';
import { Chart } from '@antv/g2';

const { LinearRegression } = datacook.Model;
const lm = async () => {
const res = await fetch('/datacook/assets/dataset/height_weight.csv');
const content = await res.text();
const data = content.split('\n').map((d) => {
  const splits = d.split(',');
  const x = parseFloat(splits[0]);
  const y = parseFloat(splits[1]);
  return { x, y };
}).slice(1);
const xData = data.map((d) => [ d.x ]);
const yData = data.map((d) => d.y);

// model train and predict
const lm = new LinearRegression();
await lm.fit(xData, yData, {batchSize: 8, epochs: 200});
const yPredict = (await lm.predict(xData)).arraySync();
const predictData = xData.map((d, i) => { return {x: d, y: yPredict[i]} });

// visualization
const chart = new G2.Chart({
  container: 'lm-chart',
  autoFit: true,
  height: 500,
  syncViewPadding: true,
});
const view1 = chart.createView(data);
view1.data(data);
view1.point().position('x*y');

const view2 = chart.createView(data);
view2.data(predictData);
view2.line().position('x*y');

chart.render();
```

<div id="lm-chart">
</div>

<script>
  const { LinearRegression } = datacook.Model;
  const lm = async () => {
    const res = await fetch('/datacook/assets/dataset/height_weight.csv');
    const content = await res.text();
    const data = content.split('\n').map((d) => {
      const splits = d.split(',');
      const height = parseFloat(splits[0]);
      const weight = parseFloat(splits[1]);
      return { height, weight };
    }).slice(1);

    const xData = data.map((d) => [ d.height ]);
    const yData = data.map((d) => d.weight);
    const lm = new LinearRegression();

    await lm.fit(xData, yData);

    const yPredict = (await lm.predict(xData)).arraySync();
    const predictData = xData.map((d, i) => { return {height: d, weight: yPredict[i]} });
    const coefs = lm.getCoef();
    coefs.coefficients.print();
    coefs.intercept.print();
    const chart = new G2.Chart({
      container: 'lm-chart',
      autoFit: true,
      height: 500,
      syncViewPadding: true,
    });

    chart.scale({
      height: {
        sync: true,
        nice: true,
      },
      weight: {
        sync: true,
        nice: true,
      },
    });

    const view1 = chart.createView(data);
    view1.data(data);
    view1.point().position('height*weight');

    const view2 = chart.createView(data);
    view2.data(predictData);
    view2.line().position('height*weight');

    chart.render();

  };
  lm();
</script>


