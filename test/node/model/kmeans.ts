import { KMeans } from '../../../src/model';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import { tensorEqual } from '../../../src/linalg/utils';
import 'mocha';
import { Tensor } from '@tensorflow/tfjs-core';

const clust1 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 3, 4 ])), tf.tensor([ 5, 5 ]));
const clust2 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 3 ])), tf.tensor([ 10, 0 ]));
const clust3 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 4, 3 ])), tf.tensor([ -10, 0 ]));
const clusData = tf.concat([ clust1, clust2, clust3 ]);

describe('KMeans', () => {
  // it('train clust', async() => {
  //   const kmeans = new KMeans({ nClusters: 3 });
  //   await kmeans.fit(clusData);
  //   const predClus = await kmeans.predict(clusData);
  //   predClus.print();
  // });
  it('train clust on batch', async() => {
    const kmeans = new KMeans({ nClusters: 3 });
    const batchSize = 30;
    const epochSize = Math.floor(clusData.shape[0] / batchSize);
    for (let i = 0; i < 1000; i++) {
      const j = Math.floor(i % epochSize);
      const batchX = tf.slice(clusData, [j * batchSize, 0], [batchSize ,2]);
      await kmeans.trainOnBatch(batchX);
    }
    kmeans.centroids.print();
    const predClus = await kmeans.predict(clusData);
    const predClusData = predClus.dataSync();
    predClus.print();
  });
});
