---
layout: default
title: standardize()
parent: Statistics
nav_order: 1
lang: en
---

# standardize()

In statistics, standardization is a widely used noramlization method. Standard score will be calculated after standardization. 

Standard score is the number of standard deviations by which the value of a raw score is above or below the mean value of what is being observed or measured. Raw scores above the mean have positive standard scores, while those below the mean have negative standard scores.

Standard score can be calculated as following:

$$\frac{X-\mu}{\sigma}$$

## Import

```javascript
import * as datacook from '@pipcook/datacook';
const { standardize } = datacook.Stat;
```

## Syntax

```javascript
standardize(xData: Tensor | RecursiveArray<number>, axis = -1): Tensor
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData    | Tensor \| RecursiveArray\<number\> | input data |
| axis `optional` | number | axis to compute, **default=-1**, which means calculation will be applied across all axes. If input data is one-dimensional, this parameter will have no effect |

## Returns

\<Tensor\> data after standardization

## Usage

```javascript
const x = [ 1, 2, 3, 4, 5 ];
const y = standardize(x);
y.print();
/**
 * Tensor
 * [-1.2649111, -0.6324555, 0, 0.6324555, 1.2649111]
 */
```



