---
layout: default
title: MLPRegressor
parent: Models
lang: en
nav_order: 9
---


# Multi-layer Perceptron Regressor

MLPRegressor is a class for fitting a regression task using multi-layer perceptron neural network

## Constructor

## Import

```javascript
import * as Datacook from 'datacook';
const { MLPRegressor } = DataCook.Model;
```

## Constructor

```typescript
const mlp = new MLPRegressor({ hiddenLayerSizes: [ 5, 6, 7 ], activations: 'relu' });
```

### Option parameters

| parameter | type | description |
| --------- | ---- | ----------- |
|  hiddenLayerSizes | number[] | size of nodes in hidden layers, length of array represents number of hidden layers in the network |
|  activations | \{ 'relu' \| 'sigmoid' \| 'tanh' \} or Array of \{ 'relu' \| 'sigmoid' \|  'tanh' \}  | activation function of hidden layers, default is **'sigmoid'** |
| optimizerType |  'sgd' \| 'momentum' \| 'adagrad' \| 'adadelta' \| 'adam' \| 'adamax' \| 'rmsprop' | optimizer types for training. All of the following [optimizers types](https://js.tensorflow.org/api/latest/#Training-Optimizers) supported in tensorflow.js can be applied. **Default to 'adam'** |
| optimizerProps | OptimizerProps | parameters used to init corresponding optimizer, you can refer to [documentations in tensorflow.js](https://js.tensorflow.org/api/latest/#Training-Optimizers) to find the supported initailization paratemters for a given type of optimizer. For example, `{ learningRate: 0.1, beta1: 0.1, beta2: 0.2, epsilon: 0.1 }` could be used to initialize adam optimizer.|  


## Methods

### fit

Fit model according to X, y.

```typescript
async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>,
    params)
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor \| RecursiveArray<number> | Tensor like of shape (n_samples, n_features), input feature |
| yData | Tensor \| RecursiveArray<number> | Tensor like of shape (n_sample, ), input target values |
| params | MLPRegressorTrainParams | option in params: - batchSize: batch size: default to 32,  - epochs: epochs for training, default to Math.ceil(10000 / ( nData / batchSize )) | 

### predict 

Make predictions using logistic regression model.

```typescript
async predict(xData: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor | RecursiveArray<number> | Input features |


### fromJson

Load model paramters from json string object

```typescript
async fromJson(modelJson: string)
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| modelJson | string | model json string |


### toJson

Dump model parameters to json string.

```typescript
async toJson(): Promise<string>
```

