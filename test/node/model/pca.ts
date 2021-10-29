
import { PCA } from '../../../src/model/pca';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import 'mocha';
import { tensorEqual } from '../../../src/linalg';

const irisData = tf.tensor2d([
  [ 5.1, 3.5, 1.4, 0.2 ],
  [ 4.9, 3., 1.4, 0.2 ],
  [ 4.7, 3.2, 1.3, 0.2 ],
  [ 4.6, 3.1, 1.5, 0.2 ],
  [ 5., 3.6, 1.4, 0.2 ],
  [ 5.4, 3.9, 1.7, 0.4 ],
  [ 4.6, 3.4, 1.4, 0.3 ],
  [ 5., 3.4, 1.5, 0.2 ],
  [ 4.4, 2.9, 1.4, 0.2 ],
  [ 4.9, 3.1, 1.5, 0.1 ],
  [ 5.4, 3.7, 1.5, 0.2 ],
  [ 4.8, 3.4, 1.6, 0.2 ],
  [ 4.8, 3., 1.4, 0.1 ],
  [ 4.3, 3., 1.1, 0.1 ],
  [ 5.8, 4., 1.2, 0.2 ],
  [ 5.7, 4.4, 1.5, 0.4 ],
  [ 5.4, 3.9, 1.3, 0.4 ],
  [ 5.1, 3.5, 1.4, 0.3 ],
  [ 5.7, 3.8, 1.7, 0.3 ],
  [ 5.1, 3.8, 1.5, 0.3 ],
  [ 5.4, 3.4, 1.7, 0.2 ],
  [ 5.1, 3.7, 1.5, 0.4 ],
  [ 4.6, 3.6, 1., 0.2 ],
  [ 5.1, 3.3, 1.7, 0.5 ],
  [ 4.8, 3.4, 1.9, 0.2 ],
  [ 5., 3., 1.6, 0.2 ],
  [ 5., 3.4, 1.6, 0.4 ],
  [ 5.2, 3.5, 1.5, 0.2 ],
  [ 5.2, 3.4, 1.4, 0.2 ],
  [ 4.7, 3.2, 1.6, 0.2 ],
  [ 4.8, 3.1, 1.6, 0.2 ],
  [ 5.4, 3.4, 1.5, 0.4 ],
  [ 5.2, 4.1, 1.5, 0.1 ],
  [ 5.5, 4.2, 1.4, 0.2 ],
  [ 4.9, 3.1, 1.5, 0.2 ],
  [ 5., 3.2, 1.2, 0.2 ],
  [ 5.5, 3.5, 1.3, 0.2 ],
  [ 4.9, 3.6, 1.4, 0.1 ],
  [ 4.4, 3., 1.3, 0.2 ],
  [ 5.1, 3.4, 1.5, 0.2 ],
  [ 5., 3.5, 1.3, 0.3 ],
  [ 4.5, 2.3, 1.3, 0.3 ],
  [ 4.4, 3.2, 1.3, 0.2 ],
  [ 5., 3.5, 1.6, 0.6 ],
  [ 5.1, 3.8, 1.9, 0.4 ],
  [ 4.8, 3., 1.4, 0.3 ],
  [ 5.1, 3.8, 1.6, 0.2 ],
  [ 4.6, 3.2, 1.4, 0.2 ],
  [ 5.3, 3.7, 1.5, 0.2 ],
  [ 5., 3.3, 1.4, 0.2 ],
  [ 7., 3.2, 4.7, 1.4 ],
  [ 6.4, 3.2, 4.5, 1.5 ],
  [ 6.9, 3.1, 4.9, 1.5 ],
  [ 5.5, 2.3, 4., 1.3 ],
  [ 6.5, 2.8, 4.6, 1.5 ],
  [ 5.7, 2.8, 4.5, 1.3 ],
  [ 6.3, 3.3, 4.7, 1.6 ],
  [ 4.9, 2.4, 3.3, 1. ],
  [ 6.6, 2.9, 4.6, 1.3 ],
  [ 5.2, 2.7, 3.9, 1.4 ],
  [ 5., 2., 3.5, 1. ],
  [ 5.9, 3., 4.2, 1.5 ],
  [ 6., 2.2, 4., 1. ],
  [ 6.1, 2.9, 4.7, 1.4 ],
  [ 5.6, 2.9, 3.6, 1.3 ],
  [ 6.7, 3.1, 4.4, 1.4 ],
  [ 5.6, 3., 4.5, 1.5 ],
  [ 5.8, 2.7, 4.1, 1. ],
  [ 6.2, 2.2, 4.5, 1.5 ],
  [ 5.6, 2.5, 3.9, 1.1 ],
  [ 5.9, 3.2, 4.8, 1.8 ],
  [ 6.1, 2.8, 4., 1.3 ],
  [ 6.3, 2.5, 4.9, 1.5 ],
  [ 6.1, 2.8, 4.7, 1.2 ],
  [ 6.4, 2.9, 4.3, 1.3 ],
  [ 6.6, 3., 4.4, 1.4 ],
  [ 6.8, 2.8, 4.8, 1.4 ],
  [ 6.7, 3., 5., 1.7 ],
  [ 6., 2.9, 4.5, 1.5 ],
  [ 5.7, 2.6, 3.5, 1. ],
  [ 5.5, 2.4, 3.8, 1.1 ],
  [ 5.5, 2.4, 3.7, 1. ],
  [ 5.8, 2.7, 3.9, 1.2 ],
  [ 6., 2.7, 5.1, 1.6 ],
  [ 5.4, 3., 4.5, 1.5 ],
  [ 6., 3.4, 4.5, 1.6 ],
  [ 6.7, 3.1, 4.7, 1.5 ],
  [ 6.3, 2.3, 4.4, 1.3 ],
  [ 5.6, 3., 4.1, 1.3 ],
  [ 5.5, 2.5, 4., 1.3 ],
  [ 5.5, 2.6, 4.4, 1.2 ],
  [ 6.1, 3., 4.6, 1.4 ],
  [ 5.8, 2.6, 4., 1.2 ],
  [ 5., 2.3, 3.3, 1. ],
  [ 5.6, 2.7, 4.2, 1.3 ],
  [ 5.7, 3., 4.2, 1.2 ],
  [ 5.7, 2.9, 4.2, 1.3 ],
  [ 6.2, 2.9, 4.3, 1.3 ],
  [ 5.1, 2.5, 3., 1.1 ],
  [ 5.7, 2.8, 4.1, 1.3 ],
  [ 6.3, 3.3, 6., 2.5 ],
  [ 5.8, 2.7, 5.1, 1.9 ],
  [ 7.1, 3., 5.9, 2.1 ],
  [ 6.3, 2.9, 5.6, 1.8 ],
  [ 6.5, 3., 5.8, 2.2 ],
  [ 7.6, 3., 6.6, 2.1 ],
  [ 4.9, 2.5, 4.5, 1.7 ],
  [ 7.3, 2.9, 6.3, 1.8 ],
  [ 6.7, 2.5, 5.8, 1.8 ],
  [ 7.2, 3.6, 6.1, 2.5 ],
  [ 6.5, 3.2, 5.1, 2. ],
  [ 6.4, 2.7, 5.3, 1.9 ],
  [ 6.8, 3., 5.5, 2.1 ],
  [ 5.7, 2.5, 5., 2. ],
  [ 5.8, 2.8, 5.1, 2.4 ],
  [ 6.4, 3.2, 5.3, 2.3 ],
  [ 6.5, 3., 5.5, 1.8 ],
  [ 7.7, 3.8, 6.7, 2.2 ],
  [ 7.7, 2.6, 6.9, 2.3 ],
  [ 6., 2.2, 5., 1.5 ],
  [ 6.9, 3.2, 5.7, 2.3 ],
  [ 5.6, 2.8, 4.9, 2. ],
  [ 7.7, 2.8, 6.7, 2. ],
  [ 6.3, 2.7, 4.9, 1.8 ],
  [ 6.7, 3.3, 5.7, 2.1 ],
  [ 7.2, 3.2, 6., 1.8 ],
  [ 6.2, 2.8, 4.8, 1.8 ],
  [ 6.1, 3., 4.9, 1.8 ],
  [ 6.4, 2.8, 5.6, 2.1 ],
  [ 7.2, 3., 5.8, 1.6 ],
  [ 7.4, 2.8, 6.1, 1.9 ],
  [ 7.9, 3.8, 6.4, 2. ],
  [ 6.4, 2.8, 5.6, 2.2 ],
  [ 6.3, 2.8, 5.1, 1.5 ],
  [ 6.1, 2.6, 5.6, 1.4 ],
  [ 7.7, 3., 6.1, 2.3 ],
  [ 6.3, 3.4, 5.6, 2.4 ],
  [ 6.4, 3.1, 5.5, 1.8 ],
  [ 6., 3., 4.8, 1.8 ],
  [ 6.9, 3.1, 5.4, 2.1 ],
  [ 6.7, 3.1, 5.6, 2.4 ],
  [ 6.9, 3.1, 5.1, 2.3 ],
  [ 5.8, 2.7, 5.1, 1.9 ],
  [ 6.8, 3.2, 5.9, 2.3 ],
  [ 6.7, 3.3, 5.7, 2.5 ],
  [ 6.7, 3., 5.2, 2.3 ],
  [ 6.3, 2.5, 5., 1.9 ],
  [ 6.5, 3., 5.2, 2. ],
  [ 6.2, 3.4, 5.4, 2.3 ],
  [ 5.9, 3., 5.1, 1.8 ]
]);

/**
 * calculated in sklearn
 * ```python
 * from sklearn.decomposition import PCA
 * from sklearn.datasets import load_iris
 * data = load_iris()
 * pca = PCA(n_components=3)
 * pca.fit(data['data'])
 * ```
 */
const irisExplainedVariance = tf.tensor([ 4.22824171, 0.24267075, 0.0782095 ]);
const irisExplainedVarianceRatio = tf.tensor([ 0.92461872, 0.05306648, 0.01710261 ]);
const irisTransformedFirst3 = tf.tensor([
  [ -2.68412563, 0.31939725, 0.02791483 ],
  [ -2.71414169, -0.17700123, 0.21046427 ],
  [ -2.88899057, -0.14494943,  -0.01790026 ]
]);

const irisExplainedVarianceCorr = tf.tensor([ 2.91808505, 0.9141649 , 0.14674182 ]);
const irisExplainedVarianceRatioCorr = tf.tensor([ 0.72962445, 0.22850762, 0.03668922 ]);
const irisTransformedFirst3Corr = tf.tensor([
  [ -2.26470281, 0.4800266 , 0.12770602 ],
  [ -2.08096115, -0.67413356, 0.23460885 ],
  [ -2.36422905, -0.34190802, -0.04420148 ]
]);

describe('Principle Component Analysis', () => {

  it('iris decomposition', async () => {
    const pca = new PCA({ nComponents: 3 });
    await pca.fit(irisData);
    console.log('explained variance:', pca.explainedVariance.arraySync());
    console.log('explained varaince ratio:', pca.explainedVarianceRatio.arraySync());
    assert.isTrue(tensorEqual(irisExplainedVarianceRatio, pca.explainedVarianceRatio, 1e-4));
    assert.isTrue(tensorEqual(irisExplainedVariance, pca.explainedVariance, 1e-4));
  });

  it('iris pca transform', async () => {
    const pca = new PCA({ nComponents: 3 });
    await pca.fit(irisData);
    const transformedData = await pca.transform(irisData);
    const transformedFirst3 = tf.slice(transformedData, [ 0, 0 ], [ 3, -1 ]);
    console.log('predicted result for first 3 samples', transformedFirst3.arraySync());
    assert.isTrue(tensorEqual(transformedFirst3, irisTransformedFirst3, 1e-4));
  });

  it('iris decomposition (correlation)', async () => {
    const pca = new PCA({ nComponents: 3, method: 'correlation' });
    await pca.fit(irisData);
    console.log('explained variance:', pca.explainedVariance.arraySync());
    console.log('explained varaince ratio:', pca.explainedVarianceRatio.arraySync());
    assert.isTrue(tensorEqual(irisExplainedVarianceRatioCorr, pca.explainedVarianceRatio, 1e-2));
    assert.isTrue(tensorEqual(irisExplainedVarianceCorr, pca.explainedVariance, 1e-2));
  });

  it('iris pca transform (correlation)', async () => {
    const pca = new PCA({ nComponents: 3, method: 'correlation' });
    await pca.fit(irisData);
    const transformedData = await pca.transform(irisData);
    const transformedFirst3 = tf.slice(transformedData, [ 0, 0 ], [ 3, -1 ]);
    console.log('predicted result for first 3 samples', transformedFirst3.arraySync());
    assert.isTrue(tensorEqual(transformedFirst3, irisTransformedFirst3Corr, 1e-2));
  });
});
