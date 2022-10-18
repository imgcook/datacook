---
layout: default
title: StandardScaler
parent: Pre-processing
nav_order: 1
lang: en
---

# StandardScaler

Standard scaler is used to standardize features by removing the mean and scaling to unit variance. The standard score of a sample `x` is calculated as: 

$$z = (x - u) / s $$

## Import

```javascript
import * as datacook from '@pipcook/datacook';
const { StandardScaler } = datacook.Preprocess;
```

## Constructor

```javascript
const standardScaler = new StandardScaler();
```

### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| params`optional` | \{ withMean?: boolean;<br/> withStd?: boolean;\} | parameters for constuctor|

#### Option in `params`

| parameter | type | description |
| --------- | ---- | ----------- |
| withMean `optional` | boolean | If true, center the data before scaling, **default=true**|
| withStd`optional` | boolean | If true, scale the data to unit variance, **default=true**


## Properties

### mean`<Tensor>`

Means of current features

### standardVariance`<Tensor>`

Standard variance of current features

### withMean `<boolean>`

Whether to center the data before scaling

### withStd `<boolean>`

Whether to scale the data to unit variance

### nFeatures `<number>`

Number of features 


## Methods

### fit()

Fit standard scaler

#### Syntax

```javascript
async fit(X: Tensor | RecursiveArray<number>): Promise<void> 
```

#### Parameters 

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor \| RecursiveArray\<number\> | input data for fitting scaler |

#### Example

```typescript
const standardScaler = new StandardScaler();
await standardScaler.fit([[0, 10], [1, 15], [2, 12]]);
```

### fitTransform()

Fit and get transformed score according to input data

#### Syntax

```typescript
async fitTransform(X: Tensor | RecursiveArray<number>): Promise<Tensor> 
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor \| RecursiveArray\<number\> | input data for fitting scaler |

#### Returns

`Promise<Tensor>`  Promise of transformed data

#### Example

```javascript
const standardScaler = new StandardScaler();
const scores= await standardScaler.fitTransform([[0, 10], [1, 15], [2, 12]]);
scores.print();
/**
 * Tensor
 *   [[-1, -0.9271726],
 *    [0 , 1.0596261 ],
 *    [1 , -0.1324531]]
 * /
```

### transform()

Get transformed score according to input data and fitted scaler.

#### Syntax

```typescript
async transform(X: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor \| RecursiveArray\<number\> | input data for getting transformed score |

#### Returns

`Promise<Tensor>`  Promise of transformed data

#### Example

```javascript
const standardScaler = new StandardScaler();
await standardScaler.fit([[0, 10], [1, 15], [2, 12]]);
const scores= await standardScaler.transform([[0, 12], [1, 10]]);
scores.print();
/**
 * Tensor
 *   [[-1, -0.1324531],
 *    [0 , -0.9271726]]
 * /
```

### inverseTransform()

Transform score to original scale.

#### Syntax

```javascript
inverseTransform(X: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor \| RecursiveArray\<number\> | input data for inverse transform |

#### Returns

`Promise<Tensor>`  Promise of data after inverse transforming


#### Example

```javascript
const standardScaler = new StandardScaler();
await standardScaler.fit([[0, 10], [1, 15], [2, 12]]);
const data= await standardScaler.inverseTransform([[1, 1], [0, 0.5]]);
data.print();
/**
 * Tensor
 *   [[2, 14.8499441],
 *    [1, 13.5916386]]
 * /
```



