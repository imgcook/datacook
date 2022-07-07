---
layout: default
title: inverse()
parent: Linear Algebra
lang: en
---
# inverse()

Inverse of a given matrix


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { inverse } = datacook.Linalg;
```

## Syntax

```typescript
async inverse(matrix: Tensor2D | number[][]): Promise<Tensor2D>
```

## parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| matrix    | Tensor | Input tensor of given matrix |

## Returns

`Promise<Tensor2D>` of inverse matrix 




