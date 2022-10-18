---
layout: default
title: getClassificationReport()
parent: Metrics
lang: en
---

# getClassificationReport()

Generate classification report.

## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getClassificationReport } = datacook.Metrics;
```

## Syntax

```typescript
getClassificationReport(yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[], average:  ClassificationAverageTypes): Promise<{ 
  precisions: Tensor;
  recalls: Tensor;
  f1s: Tensor;
  confusionMatrix: Tensor;
  categories: Tensor;
  accuracy: number;
  averagePrecision: number;
  averageRecall: number;
  averageF1: number;
}>
```


## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| yTure    | Tensor \| string[] \| number[]| True labels |
| yPred    | Tensor \| string[] \| number[]| Predicted labels |
| average  | 'macro' \| 'weighted' \| 'micro'  | |

## Returns

```typescript
Promise<{ 
  precisions: Tensor;
  recalls: Tensor;
  f1s: Tensor;
  confusionMatrix: Tensor;
  categories: Tensor;
  accuracy: number;
  averagePrecision: number;
  averageRecall: number;
  averageF1: number;
}>
```
