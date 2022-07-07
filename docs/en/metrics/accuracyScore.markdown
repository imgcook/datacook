---
layout: default
title: accuracyScore()
parent: Metrics
lang: en
---

# accuracyScore()

Get accuracy score for classification task.

In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.

## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { accuracyScore } = datacook.Metrics;
```

## Syntax

```typescript
accuracyScore(yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): number
```


## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| yTure    | Tensor | string[] | number[]| True labels |
| yPred    | Tensor | string[] | number[]| Predicted labels |

## Returns

score: `number`, the fraction of correctly classified samples 