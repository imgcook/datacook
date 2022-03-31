import * as tf from '@tensorflow/tfjs-core';
import { NonLinearRegression } from '../../../src/model/non-linear-model/NonLinearRegression';
import { assert } from 'chai';
import { Scalar } from '@tensorflow/tfjs-core';
import { numEqual } from '../../../src/math/utils';
import { tensorEqual } from '../../../src/linalg';

const nData = 100;
const cases = tf.tidy(() => tf.mul(tf.randomNormal([ nData, 5 ]), [ 1, 1, 1, 2, 3 ]));
const y = tf.tidy(() => tf.reshape(tf.sum(tf.mul(cases, 5), 1), [ -1 ]));

describe('Non-Linear Regression', () => {
  it('simple test', async () => {
    const nls = new NonLinearRegression();
    const expr = (tf: any, x: tf.Tensor, a: tf.Variable): Scalar => tf.sum(tf.mul(x, a));
    await nls.fit(expr, cases, y, { initParams: [ 3 ] });
    const coef = nls.coeffs[0].dataSync()[0];
    assert.isTrue(numEqual(coef, 5, 1e-2));
  });
  it('simple test for string input', async () => {
    const nls = new NonLinearRegression();
    const expr = '(tf, x, a) => tf.sum(tf.mul(x, a))';
    await nls.fit(expr, cases, y, { initParams: [ 3 ] });
    const coef = nls.coeffs[0].dataSync()[0];
    assert.isTrue(numEqual(coef, 5, 1e-2));
  });

  it('save and load model', async () => {
    const nls = new NonLinearRegression();
    const expr = '(tf, x, a) => tf.sum(tf.mul(x, a))';
    await nls.fit(expr, cases, y, { initParams: [ 3 ] });
    const modelJson = await nls.toJson();
    const nls2 = new NonLinearRegression();
    nls2.fromJson(modelJson);
    const yPred = await nls2.predict(cases);
    assert(tensorEqual(y, yPred, 1e-3));
  });
});
