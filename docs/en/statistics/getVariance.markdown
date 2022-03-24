---
layout: default
title: getVariance()
parent: Statistics
nav_order: 1
lang: en
---

# getVariance()

In statistics, variance is the expectation of squared deviation of a random variable.

$$ Var(X) = E[(X-\mu)^2] $$

## Import

```javascript
import * as datacook from '@pipcook/datacook';
const { getVariance } = datacook.Stat;
```

## Syntax

```javascript
getVariance(xData: Tensor | RecursiveArray<number>, axis = 0): Tensor
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData    | Tensor \| RecursiveArray\<number\> | input data |
| axis `optional` | number | axis to compute, **default=0**. If input data is one-dimensional, this parameter will have no effect |

## Usage

```javascript
const x = [ 1, 2, 3, 4, 5 ];
const y = getVariance(x);
y.print();
/**
Tensor
    2.5
**/
```





## Usage