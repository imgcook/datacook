---
layout: default
title: svd()
parent: Linear Algebra
lang: en
---

# svd()

Singular value decomposition using QR iteration and inverse iteration algorithm.

The singular value decomposition (SVD) is a factorization of a real or complex matrix. It generalizes the eigendecomposition of a square normal matrix with an orthonormal eigenbasis to any m * n matrix.

## Import 

```typescript
import * as datacook from '@pipcook/datacook';
const { svd } = datacook.Linalg;
```

## Syntax

```typescript
async svd(matrix: Tensor, tol: number, maxIter: number): Promise<Tensor>
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| matrix   | Tensor | target matrix |
| tol | number | tolerence, default to 1e-4 |
| maxIter | number | maximum iteration times, default to 200  |

## Returns

[ u, d, v ], u: left singular vector, d: singluar values, v: right singular vector
