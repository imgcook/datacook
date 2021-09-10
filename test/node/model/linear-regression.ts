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

describe('Linear Regression', () => {

  it('train simple dataset', async () => {
    const lm = new LinearRegression({optimizerType: 'adam', optimizerProps: {learningRate: 0.1}});
    await lm.fit(cases, y); 
    const { coefficients } = lm.getCoef();
    assert.isTrue(tensorEqual(coefficients, weight, 1e-1));
  });

  it('train simple dataset on batch', async () => {
    const lm = new LinearRegression({optimizerType: 'adam', optimizerProps: {learningRate: 0.1}});
    for (let i = 0; i < 800; i++) {
      const j = Math.floor(i%(100));
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
    const predY = lr2.predict(cases);
    if (predY instanceof Tensor) {
      assert.isTrue(tensorEqual(predY, y, 1));
    }
  });
});
