import { LogisticRegression } from '../../../src/model/linear-model/logistic-regression';
import { LogisticRegressionPredictor } from '../../../src/model/linear-model/logistic-regression-predictor';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import 'mocha';
import { accuracyScore } from '../../../src/metrics/classifier';
import { tensorEqual } from '../../../src/linalg/utils';

const cases = tf.mul(tf.randomNormal([ 10000, 5 ]), [ 100, 100, 200, 100, 10 ]);
const weight = tf.tensor([ 2, 3, 1, -4, 6 ]);
const eta = tf.add(tf.sum(tf.mul(cases, weight), 1), -20).arraySync() as number[];
const y = tf.greater(eta, 0);
const yMult = eta.map((e: number): number => e < -200 ? 1 : e < 0 ? 2 : e < 200 ? 3 : 4);

describe('Logistic ', () => {

  it('predict class probabilities', async () => {
    const lr = new LogisticRegression({ optimizerType: 'sgd', optimizerProps: { learningRate: 0.1 } });
    await lr.fit(cases, y);
    const probs = await lr.predictProba(cases);
    assert.equal(tf.sum(tf.lessEqual(probs, 1)).dataSync() as any, 10000);
  });

  it('predict class probabilities (multi-classification)', async () => {
    const lr = new LogisticRegression({ optimizerType: 'sgd', optimizerProps: { learningRate: 0.1 } });
    await lr.fit(cases, yMult);
    const probs = await lr.predictProba(cases);
    assert.isTrue(tensorEqual(tf.sum(probs, 1), tf.ones([ 10000 ]), 1e-5));
  });

  it('train simple dataset', async () => {
    const lr = new LogisticRegression();
    await lr.fit(cases, y);
    const predY = await lr.predict(cases);
    const acc = accuracyScore(y, predY);
    console.log('accuracy:', acc);
    assert.isTrue(acc >= 0.95);
  });

  it('train simple dataset (multi-classification)', async () => {
    const lr = new LogisticRegression();
    await lr.fit(cases, yMult);
    const predY = await lr.predict(cases);
    const acc = accuracyScore(yMult, predY);
    console.log('accuracy:', acc);
    assert.isTrue(acc >= 0.75);
  });

  it('train simple dataset on batch', async () => {
    const lr = new LogisticRegression();
    await lr.initClasses([ 0, 1 ], 'binary-only');
    const batchSize = 32;
    for (let i = 0; i < 800; i++) {
      const j = Math.floor(i % batchSize);
      const batchX = tf.slice(cases, [ j * batchSize, 0 ], [ batchSize, 5 ]);
      const batchY = tf.slice(y, [ j * batchSize ], [ batchSize ]);
      await lr.trainOnBatch(batchX, batchY);
    }
    const predY = await lr.predict(cases);
    const acc = accuracyScore(y, predY);
    console.log('accuracy:', acc);
    assert.isTrue(acc >= 0.95);
  });

  it('train simple dataset on batch (multi-classification)', async () => {
    const lr = new LogisticRegression();
    // class initialization is important for batch training
    await lr.initClasses([ 1, 2, 3, 4 ], 'binary-only');
    const batchSize = 32;
    for (let i = 0; i < 800; i++) {
      const j = Math.floor(i % batchSize);
      const batchX = tf.slice(cases, [ j * batchSize, 0 ], [ batchSize, 5 ]);
      const batchY = tf.slice(yMult, [ j * batchSize ], [ batchSize ]);
      await lr.trainOnBatch(batchX, batchY);
    }
    const predY = await lr.predict(cases);
    const acc = accuracyScore(yMult, predY);
    console.log('accuracy:', acc);
    assert.isTrue(acc >= 0.75);
  });
  it('save and load model', async () => {
    const lr = new LogisticRegression();
    await lr.fit(cases, y);
    const modelJson = await lr.toJson();
    const lr2 = new LogisticRegression();
    await lr2.fromJson(modelJson);
    const predY = await lr2.predict(cases);
    const acc = accuracyScore(y, predY);
    console.log('accuracy:', acc);
    assert.isTrue(acc >= 0.95);
  });
  it('save and load model (multi-classification)', async () => {
    const lr = new LogisticRegression();
    await lr.fit(cases, yMult);
    const modelJson = await lr.toJson();
    const lr2 = new LogisticRegression();
    await lr2.fromJson(modelJson);
    const predY = await lr2.predict(cases);
    const acc = accuracyScore(yMult, predY);
    console.log('accuracy:', acc);
    assert.isTrue(acc >= 0.75);
  });
  it('save and load model as predictor', async () => {
    const lr = new LogisticRegression();
    await lr.fit(cases, y);
    const modelJson = await lr.toJson();
    const lr2 = new LogisticRegressionPredictor();
    await lr2.fromJson(modelJson);
    const predY = await lr2.predict(cases);
    const acc = accuracyScore(y, predY);
    console.log('accuracy:', acc);
    assert.isTrue(acc >= 0.95);
  });
  it('save and load model as predictor (multi-classification)', async () => {
    const lr = new LogisticRegression();
    await lr.fit(cases, yMult);
    const modelJson = await lr.toJson();
    const lr2 = new LogisticRegressionPredictor();
    await lr2.fromJson(modelJson);
    const predY = await lr2.predict(cases);
    const acc = accuracyScore(yMult, predY);
    console.log('accuracy:', acc);
    assert.isTrue(acc >= 0.75);
  });
});
