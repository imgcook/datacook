---
layout: default
title: PCA
parent: Models
lang: en
---

# PCA (Principle Component Analysis)

Principle component analysis (PCA) is 

## Import 

```typescript
import * as Datacook from 'datacook';
const { PCA } = DataCook.Model;
```

## Constructor

```typescript
const pca = new PCA({ nComponent: 5, method: 'correlation' });
```

### Option Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| nComponent | number | the estimated number of components. If not specified, it equals the lesser value of nFeatures and nSamples | 
| method | 'covariance'\|'correlation' | Method used for decomposition. **default='covariance'**.  'covariance': use covariance matrix for decomposition. 'correlation': use correlation matrix for decomposition. |

## Methods

### fit


