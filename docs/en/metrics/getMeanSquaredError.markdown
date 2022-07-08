---
layout: default
title: getMeanSquaredError()
parent: Metrics
lang: en
---

# getRSqaure()

Computation of mean squared error.

$$
MSE = 1 / n * \sum (yTrue - yPred)^2
$$


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getMeanSquaredError } = datacook.Metrics;
```

## Syntax

```typescript
getMeanSquaredError(yTrue: Tensor1D | number[], yPred: Tensor1D | number[]): number
```


## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| yTure    | Tensor1D \| number[]| True values |
| yPred    | Tensor1D \| number[]| Predicted values |

## Returns

`number` : mean squared error
