---
layout: default
title: PCA
parent: Models
lang: en
---

# PCA (Principle Component Analysis)

Principal component analysis (PCA) is the process of computing the principal components and using them to perform a change of basis on the data, sometimes using only the first few principal components and ignoring the rest.

<img style="width:400px; max-width:100%" src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/GaussianScatterPCA.svg/600px-GaussianScatterPCA.svg.png"/>

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

Fit PCA model with input xData

#### Syntax

```typescript
async fit(xData: Tensor | RecursiveArray<number>): Promise<void>
```

#### Paramters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData     | Tensor \| RecursiveArray<number> | input data of shape (nSamples,nFeatures) in type of array or tensor |


### transform

ransform method of PCA, transform original data to reduced features.

#### Syntax

```typescript
async transform(xData: Tensor | RecursiveArray<number>): Promise<Tensor> 
```

#### Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData     | Tensor \| RecursiveArray<number> | input data of shape (nSamples,nFeatures) in type of array or tensor |

## Example

PCA of iris data

```typescript
import * as DataCook from '@pipcook/datacook';
import {PCA} from DataCook.Model;

const pca = new PCA({ nComponents: 3 });
await pca.fit(irisData);
const fittedNTensors = tf.memory().numTensors;
assert.isTrue(fittedNTensors === nTensorsBase + 5);
console.log('explained variance:', pca.explainedVariance.arraySync());
console.log('explained varaince ratio:', pca.explainedVarianceRatio.arraySync());
```
