---
layout: default
title: KMeans
parent: Models
---
# Kmeans

## Import

```javascript
import * as Datacook from 'datacook';
const { KMeans } = DataCook.Model;
```

## Constructor

```typescript
const kmeans = new KMeans({ nCluster = 5, init = 'kmeans++' });
```

### Option parameters

| parameter    | type                  | description                                                                                                                                                                                     |
| ------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| nCluster     | number                | The number of clusters to form as well as the number of centroids,**default=8**                                                                                                           |
| init         | "random"\| "kmeans++" | Centroids initialize method<br />- 'kmeans++': <br />select initial cluster centroids using kmeans++<br />- 'random': <br />randomly select initial centroids<br />**default="kmeans++"** |
| nInit        | number                | Number of time the algorithm will be run with different initialization,**default=10**                                                                                                     |
| maxIterTimes | number                | Maximum number of iterations of the k-means algorithm for a single run.**default=1000**                                                                                                   |
| `tol`      | number                | Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.**default=1e-5**                          |
| verbose      | boolean               | verbosity mode,**default=false**                                                                                                                                                          |

## Methods

### async fit(xData)

Fit kmeans model

#### Parameters

| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| xData     | Tensor\| number[][] | input data of shape (nSamples,nFeatures) in type of array or tensor |

#### Returns

tf.Tensor

### async predict(xData)

Predict sample clusters for given input.

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
|   xData   |  Tensor\| number[][]   |     input data of shape (nSamples, nFeatures) in type of array or tensor        |

#### Returns

tf.Tensor

### async score(xData)


Get scores for input xData on the kmeans model. score = -inertia, larger score usually represent better fit.

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
|   xData   |  Tensor\| number[][]   |     input data of shape (nSamples, nFeatures) in type of array or tensor        |


#### Returns
tensor of -inertia

### async trainOnBatch(xData: FeatureInputType)


Train kmeans model by batch. Here we apply mini-batch kmeans algorithm to
update centroids in each iteration. The return value is inertia copmuted for input batch.


| parameter | type | description |
| --------- | ---- | ----------- |
|   xData   |  Tensor\| number[][]   |     input data of shape (nSamples, nFeatures) in type of array or tensor        |

#### Returns

inertia for input batch data


## Examples

### Basic Usage

```javascript
import * as Datacook from 'datacook';
const { KMeans } = DataCook.Model;
const xData = [
  [1, 2], [1, 4], [1, 0],
  [10, 2], [10, 4], [10, 0]
];
const kmeans = new KMeans({ nClusters: 3 });
await kmeans.fit(xData);
const predClus = await kmeans.predict(xData);
predClus.print();
// Tensor
// [0, 0, 0, 1, 1, 1]

// save and load model
const modelJSON = await kmeans.toJson();
const kmeans2 = new KMeans({});
kmeans2.fromJson(modelJSON);
const predClus = await kmeans2.predict(xData);

predClus.print();
// Tensor
// [0, 0, 0, 1, 1, 1]

```


### Train on batch

```javascript
import * as Datacook from 'datacook';
import * as tf from '@tensorflow/tfjs-core';
const { KMeans } = DataCook.Model;

// create dataset
const clust1 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 5, 5 ]));
const clust2 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 10, 0 ]));
const clust3 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ -10, 0 ]));
const clusData = tf.concat([ clust1, clust2, clust3 ]);
// fit kmeans model
const kmeans = new KMeans({ nClusters: 3 });
const batchSize = 30;
const epochSize = Math.floor(clusData.shape[0] / batchSize);
for (let i = 0; i < 50; i++) {
   const j = Math.floor(i % epochSize);
   const batchX = tf.slice(clusData, [j * batchSize, 0], [batchSize ,2]);
   await kmeans.trainOnBatch(batchX);
}
const predClus = await kmeans.predict(clusData);
const accuracy = await checkClusAccuracy(predClus);
console.log('accuracy:', accuracy);
```
