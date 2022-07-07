---
layout: default
title: DecisionTreeClassifier
parent: Models
lang: en
nav_order: 8
---

# Decision Tree Classifier

A decision tree is a tree-like model which provide a way to present algorithms with conditional control statements. 
It includes branches that represent decision-making steps that can lead to a favorable result.

Decision tree classifier is decision tree model target for classification task.


## Constructor

## Import

```javascript
import * as Datacook from 'datacook';
const { DecisionTreeClassifier } = DataCook.Model;
```

## Constructor

```typescript
const dt = new DecisionTreeClassifier({ criterion: 'gini' });
```

### Option parameters

| parameter    | type                  | description                                                                                                                                                                                     |
| ------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `criterion` | \{"gini", "entropy"\} | The function to measure the quality of split, default="gini" |
| `maxDepth` | number | The maximum depth of the tree. If None, the nodes will expanded until all leaves are pure or until all leaves contain less than `minSamplesSplit` samples. |
| `minSampleSplit` | number | The minimum number of samples required to split an internal node: - - If integer value, then consider `minSamplesSplit` as the minimum - If float value, then `minSamplesSplit` is a fraction and `Math.ceil(minSamplesSplit * nSamples)` are the minimum number of samples for each split. |
| `minSamplesLeaf` | number | The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have qual weight when sample_weight is not provided. |
|`min_weight_fraction_leaf` | number | The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided. |
| `max_features` | number, or \{"auto", "sqrt", "log2"\} | The number of features to consider when looking for the best split: - If integer value, then consider `max_features` features at each split. - If float value, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split. - If "auto", then `max_features=sqrt(n_features)`. - If "sqrt", then `max_features=sqrt(n_features)`. - If "log2", then `max_features=log2(n_features)`. - If None, then `max_features=n_features`. Default = None |
| `ccpAlpha` | number |  Complexity parameter used for Minimal Cost-complexity Pruning. The subtree with the largest cost complexity that is smaller  than `ccpAlpha` will be chosen. By default, no pruning is performed. |


## Methods


### fit


Fit model according to X, y.

```typescript
async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>,
    params)
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor \| RecursiveArray<number> | Tensor like of shape (n_samples, n_features), input feature |
| yData | Tensor \| string[] | number[] | Tensor like of shape (n_sample, ), input target labels |
| sampleWeight | number | array | sample weights, default = null |

### predict

Make predictions using decision tree model.

```typescript
async predict(xData: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor | RecursiveArray<number> | Input features |

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

## Examples


```typescript
import * as DataCook from '@pipcook/datacook';
const { DecisionTreeClassifier } = DataCook.Model;
const dt = new DecisionTreeClassifier();
await dt.fit(irisData, labels);
const predY = await dt.predict(irisData);
const acc = accuracyScore(labels, predY);
console.log('accuracy score: ', acc);
```

