import { KMeans } from '../../../src/model';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';

const clust1 = tf.add(tf.mul(tf.randomNormal([ 20000, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 5, 5 ]));
const clust2 = tf.add(tf.mul(tf.randomNormal([ 20000, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 10, 0 ]));
const clust3 = tf.add(tf.mul(tf.randomNormal([ 20000, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ -10, 0 ]));
const clusData = tf.concat([ clust1, clust2, clust3 ]);

const checkPredTrueCnt = (predClus: tf.Tensor) => {
  return tf.max(tf.stack([
    tf.sum(tf.equal(predClus, 0)),
    tf.sum(tf.equal(predClus, 1)),
    tf.sum(tf.equal(predClus, 2))
  ])).dataSync()[0];
};

const checkClusAccuracy = async (predClus: tf.Tensor) => {
  return tf.tidy(() => {
    const predClus1 = tf.slice(predClus, 0, 100);
    const predClus2 = tf.slice(predClus, 100, 100);
    const predClus3 = tf.slice(predClus, 200, 100);
    const predTrue1 = checkPredTrueCnt(predClus1);
    const predTrue2 = checkPredTrueCnt(predClus2);
    const predTrue3 = checkPredTrueCnt(predClus3);
    return (predTrue1 + predTrue2 + predTrue3) * 1.00 / 300.0;
  });
};

describe('KMeans', () => {

  it('simple test', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const xData = [
      [ 1, 2 ], [ 1, 4 ], [ 1, 0 ],
      [ 10, 2 ], [ 10, 4 ], [ 10, 0 ]
    ];
    const kmeans = new KMeans({ nClusters: 2 });
    await kmeans.fit(xData);
    const predClus = await kmeans.predict(xData);
    predClus.print();
    predClus.dispose();
    assert.isTrue((nBaseTensors + 1) === tf.memory().numTensors);
  });

  it('train clust', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const kmeans = new KMeans({ nClusters: 3 });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
    predClus.dispose();
    assert.isTrue((nBaseTensors + 1) === tf.memory().numTensors);
  });

  it('train clust on batch', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const kmeans = new KMeans({ nClusters: 3 });
    const batchSize = 30000;
    for (let i = 0; i < 8; i++) {
      const idx = tf.tidy(() => tf.cast(tf.mul(tf.randomUniform([ batchSize ]), clusData.shape[0]), 'int32'));
      const batchX = tf.gather(clusData, idx);
      const inertia = await kmeans.trainOnBatch(batchX);
      tf.dispose([ idx, batchX ]);
      console.log(`Train with ${i}th batch, inertia ${inertia}`);
    }

    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    tf.dispose([ predClus ]);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
    // batch training will maitain two tensors: centroids and weightedSum
    assert.isTrue((nBaseTensors + 2) === tf.memory().numTensors);
  });

  it('save and load model', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const kmeans = new KMeans({ nClusters: 3 });
    await kmeans.fit(clusData);
    const modelJSON = await kmeans.toJson();
    const kmeans2 = new KMeans({});
    kmeans2.fromJson(modelJSON);
    const predClus = await kmeans2.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    predClus.dispose();
    assert.isTrue(accuracy > 0.9);
    assert.isTrue((nBaseTensors + 2) === tf.memory().numTensors);
  });

  it('different initialization (random)', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const kmeansRandom = new KMeans({ nClusters: 3, init: 'random' });
    const { selectedCentroids, inertia } = await (await kmeansRandom.initCentroids(clusData));
    console.log('inertia for random initialization (10 init times):', inertia);
    selectedCentroids.dispose();
    assert.isTrue(nBaseTensors === tf.memory().numTensors);
  });

  it('different initialization (kmeans++)', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const kmeansPP = new KMeans({ nClusters: 3 });
    const { selectedCentroids, inertia } = await (await kmeansPP.initCentroids(clusData));
    selectedCentroids.dispose();
    console.log('inertia for kmeans++ initialization (10 init times):', inertia);
    assert.isTrue(nBaseTensors === tf.memory().numTensors);
  });

  it('different initialization (user-defined)', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const initCentroids = [
      [ 0, 0 ],
      [ 1, 0 ],
      [ 4, 3 ]
    ];
    const kmeansInit = new KMeans({ nClusters: 3, init: initCentroids });
    const { selectedCentroids, inertia } = await (await kmeansInit.initCentroids(clusData));
    selectedCentroids.dispose();
    console.log('inertia for user defined initialization (10 init times):', inertia);
    assert.isTrue(nBaseTensors === tf.memory().numTensors);
  });

  it('different initialization (random, init once)', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const kmeansRandom = new KMeans({ nClusters: 3, init: 'random', nInit: 1 });
    const { selectedCentroids, inertia } = await (await kmeansRandom.initCentroids(clusData));
    console.log('inertia for random initialization (init once):', inertia);
    selectedCentroids.dispose();
    assert.isTrue(nBaseTensors === tf.memory().numTensors);
  });

  it('different initialization (kmeans++, init once)', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const kmeansPP = new KMeans({ nClusters: 3, nInit: 1 });
    const { selectedCentroids, inertia } = await (await kmeansPP.initCentroids(clusData));
    selectedCentroids.dispose();
    console.log('inertia for kmeans++ initialization (init once):', inertia);
    assert.isTrue(nBaseTensors === tf.memory().numTensors);
  });

  it('different initialization (user-defined, init once)', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const initCentroids = [
      [ 0, 0 ],
      [ 1, 0 ],
      [ 4, 3 ]
    ];
    const kmeansInit = new KMeans({ nClusters: 3, init: initCentroids, nInit: 1 });
    const { selectedCentroids, inertia } = await (await kmeansInit.initCentroids(clusData));
    selectedCentroids.dispose();
    console.log('inertia for user defined initialization (init once):', inertia);
    assert.isTrue(nBaseTensors === tf.memory().numTensors);
  });


  it('train with verbosity (kmeans++ initialization)', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const kmeans = new KMeans({ nClusters: 3, verbose: true });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    tf.dispose(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
    assert.isTrue((nBaseTensors + 1) === tf.memory().numTensors);
  });

  it('train with verbosity (random initialization)', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const kmeans = new KMeans({ nClusters: 3, verbose: true });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
    predClus.dispose();
    assert.isTrue((nBaseTensors + 1) === tf.memory().numTensors);
  });

  it('train with verbosity (user defined initialization)', async () => {
    const nBaseTensors = tf.memory().numTensors;
    const initCentroids = [
      [ 0, 0 ],
      [ 1, 0 ],
      [ 4, 3 ]
    ];
    const kmeans = new KMeans({ nClusters: 3, verbose: true, init: initCentroids });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    tf.dispose(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
    assert.isTrue((nBaseTensors + 1) === tf.memory().numTensors);
  });

});
