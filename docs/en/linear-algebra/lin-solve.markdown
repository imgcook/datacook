---
layout: default
title: linSolveQR()
parent: Linear Algebra
lang: en
---

# linSolveQR()

Solve a genral linear equation $Mx = v$ using the QR decomposition of M.

## Import 

```typescript
import * as datacook from '@pipcook/datacook';
const { linSolveFromQR } = datacook.Linalg;
```

## Syntax

```typescript
async linSolveQR(matrix: Tensor, v: Tensor): Promise<Tensor>
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| matrix   | Tensor | target matrix |
| v | Tensor | target values |

## Returns

`Promise<Tensor>`: $x$ of $Mx = v$
