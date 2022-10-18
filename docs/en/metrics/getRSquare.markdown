---
layout: default
title: getRSquare()
parent: Metrics
lang: en
---

# getRSqaure()

Computation of R-square value for regressioni task.

$$
R^2 = 1 - \frac{\sum (yTrue - yPred)^2}{\sum (yTure - \bar{yTure})^2}
$$


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getRSquare } = datacook.Metrics;
```

## Syntax

```typescript
getRSquare(yTrue: Tensor1D | number[], yPred: Tensor1D | number[]): number
```


## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| yTure    | Tensor1D \| number[]| True labels |
| yPred    | Tensor1D \| number[]| Predicted labels |

## Returns

`number` : r Square