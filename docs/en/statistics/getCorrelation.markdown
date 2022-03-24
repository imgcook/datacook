---
layout: default
title: getCorrelation()
parent: Statistics
nav_order: 1
lang: en
---

# getCorrelation()

In statistics, correlation coefficients are indicators of the strength of the linear relationship between two different variables, X and Y. A linear correlation coefficient that is greater than zero indicates a positive relationship. A value that is less thanß zero signifies a negative relationship. Finally, a value of zero indicates no relationship between the two variables X and Y.

If X and Y are two random variables, with means (expected values) $\mu X$ and $μ Y$
and standard deviations $\sigma X$ and $\sigma Y$, respectively, then their correlation is as follow:

$$  correlation = \frac{E[(X - \mu X)(Y - \mu Y)]}{\sigma X \sigma Y}  $$

## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getCorrelation } = datacook.Stat;
```

## Syntax

```typescript
getCorrelation(x: Tensor1D | number[], y: Tensor1D | number[]): number
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| x    | Tensor1D\|number[] | first input data of shape (nSamples,) in type of array or tensor |
| y    | Tensor1D\|number[] | second input data of shape (nSamples,) in type of array or tensor |

## Returns
\<number\> correlation of `x` and `y`

## Usage

```javascript
const x = [1, 4, 2, 8, 7];
const y = [2, 7, 4, 13, 10];
const corr = getCorrelation(x, y);
console.log(corr);
// 0.9899886
```
