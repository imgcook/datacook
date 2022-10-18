---
layout: default
title: KNeighborClassifier
parent: Models
lang: en
---

# KNeighborClassifier

KNeighborClassifier is 

## Import 

```javascript
import * as Datacook from 'datacook';
const { KNeighborClassifier } = DataCook.Model;
```

## Constructor

```typescript
const kNeighborClassifier = new KNeighborClassifier({ nNeighbors: 4, leafSize: 3 });
```

### Option Parameters

| parameter    | type                  | description        |
| --- | --- | --- |
| nNeighbors | number | number of neighbors for each sample |
| leafSize | number | number of samples in each tree node for BallTree or KDTree |
| weights | "uniform" \| "distance" | weights methods for generating classification. <br/>`uniform`: uniform weights;<br/> `distance`: weight points for inverse of their distance. Closer neighbor will get higher weights in this case. |
| metric | "euclidean" \| "manhattan" \| "minkowski" | metrics for computing distance. <br/> `euclidean`: euclidean distance; <br/> `manhattan`: manhattan distance; <br/> `minkowski` minkowski distance | 
| p | number | power parameter for Minkowski metric |

## Methods

### fit

#### Syntax

```typescript
async fit(xData: number[][] | Tensor2D, yData: number[] | string[] | boolean[] | Tensor1D): Promise<void>
```

#### Parameters

| Parameter | type                                              | description                                                         |
| --------- | ------------------------------------------------- | ------------------------------------------------------------------- |
| xData     | Tensor2D\| number[][]                            | input data of shape (nSamples,nFeatures) in type of array or tensor |
| yData     | Tensor1D\| number[] \| string[] \| boolean[]  | input target                                                        |

### predict

Make predictions using gradient boosting model.

```typescript
async predict(xData: Tensor|RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type   | description                 |
| --------- | ------ | --------------------------- |
| xData     | Tensor | RecursiveArray `<number>` |

#### Returns

Promise of fitted values

### fromJson

Load model paramters from json string object

```typescript
async fromJson(modelJson: string)
```

#### Parameters

| parameter | type   | description       |
| --------- | ------ | ----------------- |
| modelJson | string | model json string |

toJson

Export model paramters to json string

```typescript
async toJson(): Promise<string>
```

#### Returns

String output of model json
