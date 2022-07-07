---
layout: default
title: getJacobian()
parent: Linear Algebra
lang: en
---

# getJacobian()

Solve jacobian matrix for a given expression `expr`, input tensor `x` and varaibles

$$jac_{ij} = \frac{d(expr(x[i], var[j]))}{d(var[j])} $$


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getJacobian } = datacook.Linalg;
```

## Syntax

```typescript
getJacobian(expr: (tf: any, x: tf.Tensor, ...coeffs: tf.Variable[]) => tf.Scalar,
  x: tf.Tensor, ...coeffs: tf.Variable[]): { values: tf.Tensor, jacobian: tf.Tensor}
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| expr    | (tf: any, x: tf.Tensor, ...coeffs: tf.Variable[]) => tf.Scalar | expression function for x[i] and varaibles, return value should be a scalar
 * (expr: (tf: any, x: Tensor, ...coeffs: Variable[]) => Scalar |
| x | Tensor |  input tensor of shape [n, ....] |
| coeffs |  tf.Variable[] | array of coefficients  | 

