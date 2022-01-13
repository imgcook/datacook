---
layout: default
title: Linear Regression Analysis
parent: Models
lang: en
---

# Linear Regression Analysis

In statistics, linear regression is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables). The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression. 

Linear regression fits a linear relationship between dependent variable $y$ and one or more independent variable $X$.

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 +... + \beta_m x_m +\epsilon
$$

Compared to [LinearRegression](./linear-regression), we use ordinary least square method for model fitting instead of stochastic gradient descent. Thanks to the statistical explanatory property of OSL, we provide statistical inference and result for the linear model,like which in R.

## Import

```javascript
import * as Datacook from 'datacook';
const { LinearRegressionAnalysis } = DataCook.Model;
```

## Constructor

```typescript
lm = new LinearRegressionAnalysis();
```

### Option Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
|   fitIntercept `optional`   |  boolean  |     Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations        |
| normalize  `optional`| boolean | This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. |

## Methods

### fit

Fit linear regression model according to X, y.

```typescript
async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>,
    featureNames?: string[]): void
```

#### Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData     | Tensor \| RecursiveArray<number> | input data of shape (nSamples,nFeatures) in type of array or tensor |
| yData     | Tensor \| RecursiveArray<number> | Tensor like of shape (n_sample, ), input target values |
| featureNames `optional` | string[] | features names of shape (nFeatures, ) |


### printSummary

Print summary of model, the result will be output in console, like following.

<img src="https://img.alicdn.com/imgextra/i2/O1CN01k2Gb4928rdiVWa2sS_!!6000000007986-2-tps-1732-316.png"/>

#### Syntax

```typescript
printSummary(): void 
```

### Summary

Get summary of linear regression results.

#### Syntax

```typescript
summary(): {
    coefficients: Array<CoefficientSummary>,
    rSquare: number,
    adjustedRSquare: number,
    residualStandardError: number,
    residualDegreeOfFreedom: number,
    aic: number,
}
```

#### Returns

```typescript
{
    coefficients: Array<CoefficientSummary>,
    rSquare: number,
    adjustedRSquare: number,
    residualStandardError: number,
    residualDegreeOfFreedom: number,
    aic: number,
}
```

## Examples

Bellow is an example of fitting trees grith data using height and volumn features.

```typescript

import { Model } from '@pipcook/datacook'

const { LinearRegressionAnalysis } = Model;

const treesGrith = [ 8.3, 8.6, 8.8, 10.5, 10.7, 10.8, 11.0, 11.0, 11.1,
  11.2, 11.3, 11.4, 11.4, 11.7, 12.0, 12.9, 12.9, 13.3, 13.7, 13.8, 14.0, 14.2, 14.5,
  16.0, 16.3, 17.3, 17.5, 17.9, 18.0, 18.0, 20.6 ];
const treesHeight = [ 70, 65, 63, 72, 81, 83, 66, 75, 80, 75, 79, 76, 76,
  69, 75, 74, 85, 86, 71, 64, 78, 80, 74, 72, 77, 81, 82, 80, 80, 80, 87 ];
const treesVolumn = [ 10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6,
  19.9, 24.2, 21.0, 21.4, 21.3, 19.1, 22.2, 33.8, 27.4, 25.7, 24.9, 34.5, 31.7, 36.3,
  38.3, 42.6, 55.4, 55.7, 58.3, 51.5, 51.0, 77.0 ];

const lm = new LinearRegressionAnalysis({});
await lm.fit(cases, y);
lm.printSummary();
```