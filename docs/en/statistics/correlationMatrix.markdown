---
layout: default
title: getCorrelationMatrix()
parent: Statistics
nav_order: 1
lang: en
---

# getCorrelationMatrix()

Get covariance matrix for given input matrix with shape (n, m). If X and Y are two random variables, with means (expected values) μX and μY
and standard deviations σX and σY, respectively, then their covariance is as follow:

covariance = E[(X - μX)(Y - μY)]


## Import

```typescript
import * as datacook from '@pipcook/datacook';
const { getCovarianceMatrix } = datacook.Stat;
```

