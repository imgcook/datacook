---
layout: default
title: getPrecisionScores()
parent: Metrics
lang: en
---

# getPrecisionScores()

Compute the precision score for all classes.

Precision score is the ratio tp / (tp + fp), where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The best value is 1 and the worst value is 0.



## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getPrecisionScores } = datacook.Metrics;
```

## Syntax

```typescript
getPrecisionScores(yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): number
```


## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| yTure    | Tensor \| string[] \| number[]| True labels |
| yPred    | Tensor \| string[] \| number[]| Predicted labels |

## Returns

`Tensor` :  precision scores