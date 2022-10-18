---
layout: default
title: getAICLM()
parent: Metrics
lang: en
---

# getRSqaure()

Computation of AIC for linear model.

$$
AIC = -2 * \log L(M) + 2 * p(M)
$$

where L(M) is the likelihood of model M, p(M) is the number of indepent regressors in the model

## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getAICLM } = datacook.Metrics;
```

## Syntax

```typescript
getAICLM(yTrue: Tensor1D | number[], yPred: Tensor1D | number[], k: number): number
```


## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| yTure    | Tensor1D \| number[]| True values |
| yPred    | Tensor1D \| number[]| Predicted values |
| k | number | number of independent regressors in the model |

## Returns

`number` : aic score
