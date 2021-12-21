---
layout: default
title: Logistic Regression
parent: Models
lang: en
usemathjax: true
---

# Logistic Regression

In logistic regression model, we assume a linear relationship between predictor varaibles and log-odds of the event that $$Y = 1$$.


$$
l = \log \frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ....
$$

The objective of logistic model is to find $$\beta_0, \beta_1, \beta_2...$$ above to best fit this assumption. 

In datacook, we implement logistic regression by stochatic gradient descent (SGD). 

## Import

```javascript
import * as Datacook from 'datacook';
const { LogisticRegression } = DataCook.Model;
```

## Constructor

```typescript
const lr = new LogisticRegression({});
```

### Option parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| penalty | 'l1' \| 'l2' \| 'none' | penalty used in the penalization. **default = 'none'** |
| fitIntercept | boolean | Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations. **default = false** |
| c | number | Regularization strength; must be a positive float. Larger values specify stronger regularization. **default = 1**. |<!--  -->
| optimizerType |  'sgd' \| 'momentum' \| 'adagrad' \| 'adadelta' \| 'adam' \| 'adamax' \| 'rmsprop' | optimizer types for training. All of the following [optimizers types](https://js.tensorflow.org/api/latest/#Training-Optimizers) supported in tensorflow.js can be applied. **Default to 'adam'** |
| optimizerProps | OptimizerProps | parameters used to init corresponding optimizer, you can refer to [documentations in tensorflow.js](https://js.tensorflow.org/api/latest/#Training-Optimizers) to find the supported initailization paratemters for a given type of optimizer. For example, `{ learningRate: 0.1, beta1: 0.1, beta2: 0.2, epsilon: 0.1 }` could be used to initialize adam optimizer.|  


## Methods

### fit

Fit logistic regression model according to X, y.

```typescript
async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>,
    params?: LogisticRegressionTrainParams
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor \| RecursiveArray<number> | Tensor like of shape (n_samples, n_features), input feature |
| yData | Tensor \| RecursiveArray<number> | Tensor like of shape (n_sample, ), input target values |
| params | LogisticRegressionTrainParams | training parameters, batchSize: batch size: default to 32, maxIterTimes: max iteration times, default to 20000 | 

### predict 

Make predictions using logistic regression model.

```typescript
async predict(xData: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor | RecursiveArray<number> | Input features |

#### Returns

Predicted classes

### trainOnBatch

Training logistic regression model by batch

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor \| RecursiveArray<number> | Tensor like of shape (n_samples, n_features), input feature |
| yData | Tensor \| RecursiveArray<number> | Tensor like of shape (n_sample, ), input target values |

#### Returns

LogisticRegression


### predictProba

Predict probabilities using logistic regression model.

```typescript
async predictProba(xData: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor \| RecursiveArray<number> | Input features|

#### Returns

Predicted probabilities

### getCoef

Get coefficients of logistic model

```typescript
getCoef(): { coefficients: Tensor, intercept: Tensor }
```
#### Returns

{ coefficients: Tensor, intercept: Tensor }

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

#### Returns

string of model json


## Examples

```typescript

```


