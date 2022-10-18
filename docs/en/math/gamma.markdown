---
layout: default
title: gamma()
parent: Math
nav_order: 6
lang: en
---
# gamma()

In mathematics, the gamma function (represented by Γ, the capital letter gamma from the Greek alphabet) is one commonly used extension of the factorial function to complex numbers. The gamma function is defined for all complex numbers except the non-positive integers. For any positive integer n,

$$
\Gamma(n) = (n - 1)!
$$


For complex numbers with a positive real part, the gamma function is defined via a convergent improper integral:

$$
\Gamma(z) = \int_0^\inf x^{z-1}e^{-x} dx
$$


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { gamma } = datacook.Math;
```

## Syntax

```typescript
gamma(x: number): number 
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| x    | number | $x$ of $\Gamma(x)$ | 








 

