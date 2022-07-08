---
layout: default
title: getRecallScores()
parent: Metrics
lang: en
---

# getRecallScores()

Compute the precision score for all classes.

Recall score is the ratio tp / (tp + fn), where tp is the number of true positives and fn the number of false negtive. The recall is intuitively the ability of the classifier to find all the positive samples.


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getRecallScores } = datacook.Metrics;
```

## Syntax

```typescript
getRecallScores(yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): number
```


## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| yTure    | Tensor \| string[] \| number[]| True labels |
| yPred    | Tensor \| string[] \| number[]| Predicted labels |

## Returns

`Tensor` :  tensor of recall scores