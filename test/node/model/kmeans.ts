import { KMeans } from '../../../src/model';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import 'mocha';
import { assert } from 'chai';

const clust1 = tf.add(tf.mul(tf.randomNormal([ 200000, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 5, 5 ]));
const clust2 = tf.add(tf.mul(tf.randomNormal([ 200000, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 10, 0 ]));
const clust3 = tf.add(tf.mul(tf.randomNormal([ 200000, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ -10, 0 ]));
const clusData = tf.concat([ clust1, clust2, clust3 ]);

const checkPredTrueCnt = (predClus: tf.Tensor) => {
  return tf.max(tf.stack([
    tf.sum(tf.equal(predClus, 0)), 
    tf.sum(tf.equal(predClus, 1)), 
    tf.sum(tf.equal(predClus, 2))
  ])).dataSync()[0];
};

const checkClusAccuracy = async (predClus: tf.Tensor) => {
  const predClus1 = tf.slice(predClus, 0, 100);
  const predClus2 = tf.slice(predClus, 100, 100);
  const predClus3 = tf.slice(predClus, 200, 100);
  const predTrue1 = checkPredTrueCnt(predClus1);
  const predTrue2 = checkPredTrueCnt(predClus2);
  const predTrue3 = checkPredTrueCnt(predClus3);
  return (predTrue1 + predTrue2 + predTrue3) * 1.00 / 300.0;
};

describe('KMeans', () => {

  it('simple test', async ()=> {
    const xData = [
      [1, 2], [1, 4], [1, 0],
      [10, 2], [10, 4], [10, 0]
    ];
    const kmeans = new KMeans({ nClusters: 2 });
    await kmeans.fit(xData);
    const predClus = await kmeans.predict(xData);
    predClus.print();
  })

  it('train clust', async () => {
    const kmeans = new KMeans({ nClusters: 3, init: 'random', verbose: true });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
  });

  it('train clust on batch', async () => {
    const kmeans = new KMeans({ nClusters: 3 });
    const batchSize = 500;
    for (let i = 0; i < 100; i++) {
      const idx = tf.cast(tf.mul(tf.randomUniform([batchSize]), clusData.shape[0]),'int32');
      const batchX = tf.gather(clusData, idx);
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
    const kmeans2 = new KMeans({});
    kmeans2.fromJson(modelJSON);
    const predClus = await kmeans2.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
  });
  
  it('different initialization', async () => {
    const initCentroids = [
      [ 0, 0 ],
      [ 1, 0 ],
      [ 4, 3 ]
    ];
    const kmeansRandom = new KMeans({ nClusters: 3, init: 'random' });
    const inertiaRandom = await (await kmeansRandom.initCentroids(clusData)).inertia;
    console.log('inertia for random initialization (10 init times):', inertiaRandom);

    const kmeansPP = new KMeans({ nClusters: 3 });
    const inertiaPP = await (await kmeansPP.initCentroids(clusData)).inertia;
    console.log('inertia for kmeans++ initialization (10 init times):', inertiaPP);

    const kmeansRandom1 = new KMeans({ nClusters: 3, init: 'random', nInit: 1 });
    const inertiaRandom1 = await (await kmeansRandom1.initCentroids(clusData)).inertia;
    console.log('inertia for random initialization (1 init times):', inertiaRandom1);
    const kmeansPP1 = new KMeans({ nClusters: 3, nInit: 1 });
    const inertiaPP1 = await (await kmeansPP1.initCentroids(clusData)).inertia;
    console.log('inertia for kmeans++ initialization (1 init times):', inertiaPP1);

    const kmeansInit = new KMeans({ nClusters: 3, init: initCentroids });
    const inertiaInit = await (await kmeansInit.initCentroids(clusData)).inertia;
    console.log('inertia for user defined initialization:', inertiaInit);
  });

  it('train with verbosity (kmeans++ initialization)', async () => {
    const kmeans = new KMeans({ nClusters: 3, verbose: true });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
  });

  it('train with verbosity (random initialization)', async () => {
    const kmeans = new KMeans({ nClusters: 3, verbose: true });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
  });

  it('train with verbosity (user defined initialization)', async () => {
    const initCentroids = [
      [ 0, 0 ],
      [ 1, 0 ],
      [ 4, 3 ]
    ];
    const kmeans = new KMeans({ nClusters: 3, verbose: true, init: initCentroids });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    const accuracy = await checkClusAccuracy(predClus);
    console.log('accuracy:', accuracy);
    assert.isTrue(accuracy > 0.9);
  });

});
