---
layout: default
title: getF1Scores()
parent: Metrics
lang: en
---

# getF1Scores()

Compute the f1 score for all classes.

 The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is: `2 * precision * recall / (precision + recall)`


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getF1Scores } = datacook.Metrics;
```

## Syntax

```typescript
getF1Scores(yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): number
```


## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| yTure    | Tensor \| string[] \| number[]| True labels |
| yPred    | Tensor \| string[] \| number[]| Predicted labels |

## Returns

`Tensor` :  f1 scores