---
layout: default
title: beta()
parent: Math
nav_order: 6
lang: en
---
# beta()

Computation of beta function. Beta function, also called the Euler integral of the first kind, is a special function that is closely related to the gamma function and to binomial coefficients. It is defined by the integral

$$
B(x, y) = \int_0^1 t^{x-1}(1-t)^{y-1}dt
$$

## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { beta } = datacook.Math;
```

## Syntax

```typescript
beta(x: number, y: number): number 
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| x    | number | First shape parameter |
| y    | number | Second shape parameter |




 

