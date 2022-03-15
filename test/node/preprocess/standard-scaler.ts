import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { assert } from 'chai';
import 'mocha';
import { StandardScaler } from '../../../src/preprocess';
import { getVariance } from '../../../src/stat';


const cases = tf.mul(tf.sub(tf.randomNormal([ 10000, 5 ]), [ 5, 3, 2, 1, -5 ]), [ 100, 100, 200, 100, 10 ]);

describe('Standard Scaler', () => {
  it('fit', async () => {
    const scaler = new StandardScaler();
    await scaler.fit(cases);
    const transformed = await scaler.transform(cases);
    const means = tf.mean(transformed, 0);
    const variance = getVariance(transformed);
    assert.isTrue(tensorEqual(means, tf.zeros([ 5 ]), 1e-3));
    assert.isTrue(tensorEqual(variance, tf.ones([ 5 ]), 1e-3));
  });
  it('save and load model', async () => {
    const scaler = new StandardScaler();
    await scaler.fit(cases);
    const modelJson = await scaler.toJson();
    const scaler2 = new StandardScaler();
    await scaler2.fromJson(modelJson);
    const transformed = await scaler2.transform(cases);
    const means = tf.mean(transformed, 0);
    const variance = getVariance(transformed);
    assert.isTrue(tensorEqual(means, tf.zeros([ 5 ]), 1e-3));
    assert.isTrue(tensorEqual(variance, tf.ones([ 5 ]), 1e-3));
  });
});
