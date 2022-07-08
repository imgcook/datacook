---
layout: default
title: getRedisualVariance()
parent: Metrics
lang: en
---

# getRSqaure()

Computation of residual variance.

$$
Var(residual) = 1 /(n - df(model)) * \sum (yTrue - yPred)^2 
$$


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getResidualVariance } = datacook.Metrics;
```

## Syntax

```typescript
getResidualVariance(yTrue: Tensor1D | number[], yPred: Tensor1D | number[]): number
```


## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| yTure    | Tensor1D \| number[]| True values |
| yPred    | Tensor1D \| number[]| Predicted values |

## Returns

`number` : residual variance
