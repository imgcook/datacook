---
layout: default
title: getHistData()
parent: Statistics
nav_order: 1
lang: en
---

# getHistData()

Get histogram data for given input

## Import 

```typescript
import * as datacook from '@pipcook/datacook';
const { getHistData } = datacook.Stat;
```

## Syntax

```typescript
 getHistData(xData: Tensor1D | number[], params?: {
    bins?: number,
    leftLimit?: number,
    rightLimit?: number
  }): {
    steps: number[],
    counts: number[],
  }
```

## Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData    | Tensor1D \| number [] | input data of shape [nSamples, ] |
| params | { bins?: number, leftLimit?: number, rightLimit?: number } | parameters |

### Option in params


- `bins`: number of cells, **default=50**
- `leftLimit`: left limit to calculate, **default=min value of input data**
- `rightLimit`: right limit to calculate, **default=max value of input data**


## Returns 

 `{ steps; number[], counts: number[] }`
 * `steps`:  array of giving the breakpoints between histogram cells,
 * `counts`:  array of number of data points in histogram cells