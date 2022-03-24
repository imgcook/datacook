---
layout: default
title: getCovariance()
parent: Statistics
nav_order: 1
lang: en
---

# getCovariance()

In statistics, covariance is a measure of the relationship between two random variables. The metric evaluates how much, or to what extent – the variables change together. 

If X and Y are two random variables, with means (expected values) $\mu X$ and $\mu Y$, their covariance is as follow:

$$covariance = E[(X - \mu X)(Y - \mu Y)]$$


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getCovariance } = datacook.Stat;
```

## Syntax

```typescript
getCovariance(x: Tensor1D | number[], y: Tensor1D | number[]): number
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| x    | Tensor1D\|number[] | first input data of shape (nSamples,) in type of array or tensor |
| y    | Tensor1D\|number[] | second input data of shape (nSamples,) in type of array or tensor |

## Returns
\<number\> covariance of `x` and `y`

## Usage

```javascript
const x = [5, 10, 2, 4, 2];
const y = [2, 8, 7, 6, 1];
const cov = getCovariance(x, y);
console.log(cov);
// 4.9
```




