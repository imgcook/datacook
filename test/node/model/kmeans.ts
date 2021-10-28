import { KMeans } from '../../../src/model';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import 'mocha';
import { assert } from 'chai';
import { Tensor } from '@tensorflow/tfjs-core';

const clust1 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 5, 5 ]));
const clust2 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 10, 0 ]));
const clust3 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ -10, 0 ]));
const clusData = tf.concat([ clust1, clust2, clust3 ]);

const checkPredTrueCnt = (predClus: Tensor) => {
  return tf.max(tf.stack([tf.sum(tf.equal(predClus, 0)), tf.sum(tf.equal(predClus, 1)), tf.sum(tf.equal(predClus, 2))])).dataSync()[0];
}

const checkClusAccuracy = async (predClus: Tensor) => {
  const predClus1 = tf.slice(predClus, 0, 100);
  const predClus2 = tf.slice(predClus, 100, 100);
  const predClus3 = tf.slice(predClus, 200, 100);
  const predTrue1 = checkPredTrueCnt(predClus1);
  const predTrue2 = checkPredTrueCnt(predClus2);
  const predTrue3 = checkPredTrueCnt(predClus3);
  return (predTrue1 + predTrue2 + predTrue3) * 1.00 / 300.0;
}

describe('KMeans', () => {
  it('train clust', async () => {
    const kmeans = new KMeans({ nClusters: 3 });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
  });
  it('train clust on batch', async () => {
    const kmeans = new KMeans({ nClusters: 3 });
    const batchSize = 30;
    const epochSize = Math.floor(clusData.shape[0] / batchSize);
    for (let i = 0; i < 1000; i++) {
      const j = Math.floor(i % epochSize);
      const batchX = tf.slice(clusData, [j * batchSize, 0], [batchSize ,2]);
      await kmeans.trainOnBatch(batchX);
    }
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
  });
  it('save and load model', async () => {
    const kmeans = new KMeans({ nClusters: 3 });
    await kmeans.fit(clusData);
    const modelJSON = await kmeans.toJson();
    const kmeans2 = new KMeans({ nClusters: 3 });
    kmeans2.fromJson(modelJSON);
    const predClus = await kmeans2.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
  })
});
