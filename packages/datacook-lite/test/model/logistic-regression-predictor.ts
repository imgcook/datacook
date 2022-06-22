import 'module-alias/register';
import { assert } from 'chai';
import { LogisticRegression } from '@pipcook/datacook/dist/model/linear-model/logistic-regression';
import { LogisticRegressionPredictor } from '../../src/model/linear-model/logistic-regression-predictor';
import * as tf from '@tensorflow/tfjs-core';
import { accuracyScore } from '../../src/metrics/classification';

const cases = tf.mul(tf.randomNormal([ 10000, 5 ]), [ 100, 100, 200, 100, 10 ]);
const weight = tf.tensor([ 2, 3, 1, -4, 6 ]);
const eta = tf.add(tf.sum(tf.mul(cases, weight), 1), -20).arraySync() as number[];
const y = tf.greater(eta, 0);
const yMult = eta.map((e: number): number => e < -200 ? 1 : e < 0 ? 2 : e < 200 ? 3 : 4);

describe('Logistic ', () => {

  it('save and load model', async () => {
    const lr = new LogisticRegression({ optimizerType: 'sgd', optimizerProps: { learningRate: 0.1 } });
    await lr.fit(cases, y);
    const modelJson = await lr.toJson();
    const lr2 = new LogisticRegressionPredictor();
    await lr2.fromJson(modelJson);
    const predY = await lr2.predict(cases.arraySync() as number[][]);
    const predYOrigin = await lr.predict(cases.arraySync() as number[][]);
    const acc = accuracyScore(y.arraySync() as number[], predY);
    const accOrigin = accuracyScore(y.arraySync() as number[], predYOrigin.arraySync() as number[]);
    console.log('accuracy:', acc);
    console.log('accuracyOrigin:', accOrigin);
    assert.isTrue(acc === accOrigin);
  });

  it('save and load model (multi-classification)', async () => {
    const lr = new LogisticRegression({ optimizerType: 'sgd', optimizerProps: { learningRate: 0.1 } });
    await lr.fit(cases, yMult);
    const modelJson = await lr.toJson();
    const lr2 = new LogisticRegressionPredictor();
    await lr2.fromJson(modelJson);
    const predY = await lr2.predict(cases.arraySync() as number[][]);
    const predYOrigin = await lr.predict(cases.arraySync() as number[][]);
    const acc = accuracyScore(yMult, predY);
    const accOrigin = accuracyScore(yMult, predYOrigin.arraySync() as number[]);
    console.log('accuracy:', acc);
    console.log('accuracyOrigin:', accOrigin);
    assert.isTrue(acc === accOrigin);
  });
});
