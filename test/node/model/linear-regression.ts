import { LinearRegression } from '../../../src/model/linear-model/linear-regression';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import { tensorEqual } from '../../../src/linalg/utils';
import 'mocha';
import { Tensor } from '@tensorflow/tfjs-core';

const cases = tf.mul(tf.randomNormal([10000, 5]),[ 1, 10, 100, 2, 3 ]);
const weight = tf.tensor([ 2, 3, 1, 4, 6 ])
const y = tf.add(tf.sum(tf.mul(cases, weight), 1), 10);

const heights = [
  1.47,
  1.5,
  1.52,
  1.55,
  1.57,
  1.6,
  1.63,
  1.65,
  1.68,
  1.7,
  1.73,
  1.75,
  1.78,
  1.8,
  1.83
];

const weights = [
  52.21,
  53.12,
  54.48,
  55.84,
  57.2,
  58.57,
  59.93,
  61.29,
  63.11,
  64.47,
  66.28,
  68.1,
  69.92,
  72.19,
  74.46
];


describe('Linear Regression', () => {

  it('train on weight and height', async () => {
    const lm = new LinearRegression();
    await lm.fit(heights.map((d) => [d]), weights);
    const yPred = await lm.predict(heights.map((d) => [d]));
    const meanDist = tf.tidy(() => tf.sum(tf.sub(tf.mean(yPred), tf.tensor(weights))).dataSync()[0]);
    assert.isTrue(meanDist < 0.1);
  });

  it('train simple dataset', async () => {
    const lm = new LinearRegression({optimizerType: 'adam', optimizerProps: {learningRate: 0.1}});
    await lm.fit(cases, y); 
    const { coefficients } = lm.getCoef();
    assert.isTrue(tensorEqual(coefficients, weight, 1e-1));
  });

  it('train simple dataset on batch', async () => {
    const lm = new LinearRegression({optimizerType: 'adam', optimizerProps: {learningRate: 0.1}});
    for (let i = 0; i < 800; i++) {
      const j = Math.floor(i % 100);
      const batchX = tf.slice(cases, [j * 100, 0], [100 ,5]);
      const batchY = tf.slice(y, [j * 100], [100]);
      await lm.trainOnBatch(batchX, batchY)
    }
    const { coefficients } = lm.getCoef();
    assert.isTrue(tensorEqual(coefficients, weight, 1e-1));
  });

  it('save and load model', async () => {
    const lr = new LinearRegression({fitIntercept: true});
    await lr.fit(cases, y); 
    const modelJson = await lr.toJson();
    const lr2 = new LinearRegression();
    await lr2.fromJson(modelJson);
    const predY = await lr2.predict(cases);
    assert.isTrue(tensorEqual(predY, y, 1));
  });
});
