---
layout: default
title: Linear Regression
parent: Models
lang: en
---

# Linear Regression

LinearRegression fits a linear model with coefficients w = (w1, …, wp) 
to minimize the residual sum of squares between the observed targets in
the dataset, and the targets predicted by the linear approximation.

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
|   fitIntercept   |  boolean  |     Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations        |
| normalize | boolean | This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. |
| optimizerType |  'sgd' \| 'momentum' \| 'adagrad' \| 'adadelta' \| 'adam' \| 'adamax' \| 'rmsprop' | optimizer types for training. All of the following [optimizers types](https://js.tensorflow.org/api/latest/#Training-Optimizers) supported in tensorflow.js can be applied. **Default to 'adam'** |
| optimizerProps | OptimizerProps | parameters used to init corresponding optimizer, you can refer to [documentations in tensorflow.js](https://js.tensorflow.org/api/latest/#Training-Optimizers) to find the supported initailization paratemters for a given type of optimizer. For example, `{ learningRate: 0.1, beta1: 0.1, beta2: 0.2, epsilon: 0.1 }` could be used to initialize adam optimizer.|  


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
| params   | LinearRegerssionTrainParams | `batchSize`: batch size: **default = 32**, `maxIterTimes`: max iteration times, **default = 20000** |

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


