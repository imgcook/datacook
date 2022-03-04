import { KMeans } from '../../../src/model';
import { getSilhouetteCoefficient } from '../../../src/metrics/clustering';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';

const clust1 = tf.add(tf.mul(tf.randomNormal([ 2000, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 5, 5 ]));
const clust2 = tf.add(tf.mul(tf.randomNormal([ 2000, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 10, 0 ]));
const clust3 = tf.add(tf.mul(tf.randomNormal([ 2000, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ -10, 0 ]));
const clusData = tf.concat([ clust1, clust2, clust3 ]);

describe('Clustering metrics', () => {
  it('calculate elbow data', async () => {
    const kmeans = new KMeans({ nClusters: 3 });
    await kmeans.fit(clusData);
    const predClus = await kmeans.predict(clusData);
    console.log('fitted');
    const coeffs = await getSilhouetteCoefficient(clusData, predClus);
    console.log(coeffs);
    // const elbowData = await kElbow(kmeans, clusData, { verbose: true });
    // console.log(elbowData);
    // console.log(kmeans.nClusters);
  });
});
