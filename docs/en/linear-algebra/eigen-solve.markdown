---
layout: default
title: eigenSolve()
parent: Linear Algebra
lang: en
---

# eigenSolve()

Eigen decomposition of given matrix

## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { eigenSolve } = datacook.Linalg;
```

## Syntax

```typescript
async eigenSolve(matrix: Tensor, tol: number, maxIter: number): Promise<[Tensor, Tensor]>
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| matrix    | Tensor | Input tensor of given matrix |
| tol | number | stop tolerence, default to 1e-4 | 
| maxIter | number | maximum iteration times, default to 200 |

## Return

`Promise<[Tensor, Tensor]>` Tensor array of [ eigenValues, eigenVectors ]



