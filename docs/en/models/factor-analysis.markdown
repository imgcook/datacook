---
layout: default
title: Factor Analysis
parent: Models
lang: en
nav_order: 8
---

# Factor Analysis

Factor analysis is a technique that is used to reduce a large number of variables into fewer numbers of factors.  This technique extracts maximum common variance from all variables and puts them into a common score.  As an index of all variables, we can use this score for further analysis. 

## Import 


```javascript
import * as Datacook from 'datacook';
const { FactorAnalysis } = DataCook.Model;
```

## Constructor

```typescript
const fa = new FactorAnalysis({ nComponent: 5 });
```

### Option parameters



| parameter    | type                   |description                                                                               |
| ------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| nComponents | number | number of components for decomposition |
| tol  | number | tolenrence for iteration |
| maxIterTimes  | number | maximum iteration times |

## Properties

### nComponents

`number` : number of compoennts

### facorLoadings

`Tensor` :  factor loadings after decompsition

## Methods

### fit

Fit factor analysis model according to given dataset.

#### Syntax

```typescript
async fit(xData: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor \| RecursiveArray<number> | input dataset |


#### Returns

`Tensor` : Factor loadings for given dataset



### fromJson

Load model paramters from json string object

```typescript
async fromJson(modelJson: string)
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| modelJson | string | model json string |

### toJson

Export model paramters to json string

```typescript
async toJson(): Promise<string>
```

#### Returns

String output of model json


