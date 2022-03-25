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
| axis `optional` | number | axis to compute, **default=-1**, which means calculation will be applied across all axes. If input data is one-dimensional, this parameter will have no effect |

## Usage

Variance calculation for one-dimensinoal data:
```javascript
const x = [ 1, 2, 3, 4, 5 ];
const y = getVariance(x);
y.print();
/**
Tensor
    2.5
**/
```

Variance calculation for two-dimensinoal data:
```javascript
const x =  [ [ 1, 2, 3, 4, 5 ], [ 6, 7, 8, 9, 10 ] ];
const v1 = getVariance(x, 0);
v1.print();
/**
 * Tensor
 * [12.5, 12.5, 12.5, 12.5, 12.5]
 ** /
```

```javascript
const x =  [ [ 1, 2, 3, 4, 5 ], [ 6, 7, 8, 9, 10 ] ];
const v2 = getVariance(x, 1);
v2.print();
/**
 * Tensor
 * [2.5, 2.5]
 ** /
```