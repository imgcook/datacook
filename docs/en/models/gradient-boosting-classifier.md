---
layout: default

title: GradientBoostingClassifier

parent: Models

lang: en

nav_order: 8
---
# GradientBoostingClassifier

Gradient boosting is a machine learning method which provides predictions by training an ensemble of weak estimatorss. GradientBoostingClassifier is an implementation of gradient boosting for classification task.

## Import

```typescript
import * as DataCook from '@pipcook/datacook';
const { GradientBoostingClassifier } = DataCook.Model;
```

## Constructor

```typescript
const gb = newGradientBoostingClassifier({ nEstimators: 10 });
```

### Option parameters

| Parameter             | Type                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| nEstimators           | number                              | number of estimators for fitting. default = 100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| criterion             |                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| minSamplesLeaf        | number                              | The minimum number of samples required to be at leaf node, default = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| minImpurityDecrease   | number                              | A node will be split if this split induces a decrease of the impurity greater than or equal to this value                                                                                                                                                                                                                                                                                                                                                                                                                               |
| minWeightFractionLeaf |                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| minSamplesSplit       | number                              | Minimum number of samples required to split an internal node, default = 2va                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| validationFraction    | number                              | The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if `nIterNoChange` is set to an integer.                                                                                                                                                                                                                                                                                                                                                                      |
| ccpAlpha              | number                              | Complexity parameter used for Minimal Cost-complexity Pruning. The subtree with the largest cost complexity that is smaller than `ccpAlpha` will be chosen. By default, no pruning is performed.                                                                                                                                                                                                                                                                                                                                   |
| maxDepth              | number                              | Maximum depth of the individual regression tree, default = 3                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| maxFeatures           | number, or {"auto", "sqrt", "log2"} | The number of features to consider when looking for the best split:<br />- If integer value, then consider `maxFeatures` features at each split.<br />- If not interger value, then `maxFeatures` is a fraction and `Math.floor(maxFeatures * nFeatures)` features are considered at each split.<br />- If "auto", then `max_features=sqrt(n_features)`.<br />- If "sqrt", then `maxFeatures=sqrt(nFeatures)`.<br />- If "log2", then `maxFeatures=log2(nFeatures)`.<br />- If none, then `maxFeatures=nFatures`. |
| tol                   | number                              | Tolerance for the early stopping. When the loss is not improving by at least tol for ``nIterNoChange`` iterations (if set to a number), the training stops, default = 1e-4                                                                                                                                                                                                                                                                                                                                                            |
| nIterNoChange         | number                              | Used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to `none` to disable early stopping. If set to a number, it will set aside `validation_fraction` size of the training data as validation and terminate training when validation score is not improving in all of the previous `nIterNoChange` numbers of iterations. The split is stratified.                                                                                              |

## Methods

### fit

Fit gradient boosting classifier

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
