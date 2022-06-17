import 'module-alias/register';
import { assert } from 'chai';
import { LogisticRegression } from '@pipcook/datacook/dist/model/linear-model/logistic-regression';
import { LogisticRegressionPredictor } from '../../src/model/linear-model/logistic-regression-predictor';
import * as tf from '@tensorflow/tfjs-core';
import { accuracyScore } from '../../src/metrics/classification';


// import { add,  }
const cases = tf.mul(tf.randomNormal([ 10000, 5 ]), [ 100, 100, 200, 100, 10 ]);
const weight = tf.tensor([ 2, 3, 1, -4, 6 ]);
const eta = tf.add(tf.sum(tf.mul(cases, weight), 1), -20).arraySync() as number[];
const y = tf.greater(eta, 0);
const yMult = eta.map((e: number): number => e < -200 ? 1 : e < 0 ? 2 : e < 200 ? 3 : 4);


// const cases: number[][] = [];
// for (let i = 0; i < 10000; i++) {
//   cases.push([ Math.random() * 100, Math.random() * 100, Math.random() * 200, Math.random() * 100, Math.random() * 10 ]);
// }

// const weight: number[] = [ 2, 3, 1, -4, 6 ];
// // const eta =  tf.add(tf.sum(tf.mul(cases, weight), 1), -20).arraySync() as number[];
// // const eta = add2d(sum2d(mul2d(matrix(cases), vector(weight)), 1), -20);
// const eta = (sum2d(mul2d(matrix(cases), vector(weight)), 1) as Vector).data.map((d) => d - 20);
// const y = eta.map((d) => d > 0);
// const yMult = eta.map((e: number): number => e < -200 ? 1 : e < 0 ? 2 : e < 200 ? 3 : 4);

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

  // it('predict class probabilities (multi-classification)', async () => {
  //   const lr = new LogisticRegression({ optimizerType: 'sgd', optimizerProps: { learningRate: 0.1 } });
  //   await lr.fit(cases, yMult);
  //   const probs = await lr.predictProba(cases);
  //   assert.isTrue(tensorEqual(tf.sum(probs, 1), tf.ones([ 10000 ]), 1e-5));
  // });

  // it('train simple dataset', async () => {
  //   const lr = new LogisticRegression();
  //   await lr.fit(cases, y);
  //   const predY = await lr.predict(cases);
  //   const acc = accuracyScore(y, predY);
  //   console.log('accuracy:', acc);
  //   assert.isTrue(acc >= 0.95);
  // });

  // it('train simple dataset (multi-classification)', async () => {
  //   const lr = new LogisticRegression();
  //   await lr.fit(cases, yMult);
  //   const predY = await lr.predict(cases);
  //   const acc = accuracyScore(yMult, predY);
  //   console.log('accuracy:', acc);
  //   assert.isTrue(acc >= 0.75);
  // });

  // it('train simple dataset on batch', async () => {
  //   const lr = new LogisticRegression();
  //   await lr.initClasses([ 0, 1 ], 'binary-only');
  //   const batchSize = 32;
  //   for (let i = 0; i < 800; i++) {
  //     const j = Math.floor(i % batchSize);
  //     const batchX = tf.slice(cases, [ j * batchSize, 0 ], [ batchSize, 5 ]);
  //     const batchY = tf.slice(y, [ j * batchSize ], [ batchSize ]);
  //     await lr.trainOnBatch(batchX, batchY);
  //   }
  //   const predY = await lr.predict(cases);
  //   const acc = accuracyScore(y, predY);
  //   console.log('accuracy:', acc);
  //   assert.isTrue(acc >= 0.95);
  // });

  // it('train simple dataset on batch (multi-classification)', async () => {
  //   const lr = new LogisticRegression();
  //   // class initialization is important for batch training
  //   await lr.initClasses([ 1, 2, 3, 4 ], 'binary-only');
  //   const batchSize = 32;
  //   for (let i = 0; i < 800; i++) {
  //     const j = Math.floor(i % batchSize);
  //     const batchX = tf.slice(cases, [ j * batchSize, 0 ], [ batchSize, 5 ]);
  //     const batchY = tf.slice(yMult, [ j * batchSize ], [ batchSize ]);
  //     await lr.trainOnBatch(batchX, batchY);
  //   }
  //   const predY = await lr.predict(cases);
  //   const acc = accuracyScore(yMult, predY);
  //   console.log('accuracy:', acc);
  //   assert.isTrue(acc >= 0.75);
  // });
  // it('save and load model', async () => {
  //   const lr = new LogisticRegression();
  //   await lr.fit(cases, y);
  //   const modelJson = await lr.toJson();
  //   const lr2 = new LogisticRegression();
  //   await lr2.fromJson(modelJson);
  //   const predY = await lr2.predict(cases);
  //   const acc = accuracyScore(y, predY);
  //   console.log('accuracy:', acc);
  //   assert.isTrue(acc >= 0.95);
  // });
  // it('save and load model (multi-classification)', async () => {
  //   const lr = new LogisticRegression();
  //   await lr.fit(cases, yMult);
  //   const modelJson = await lr.toJson();
  //   const lr2 = new LogisticRegression();
  //   await lr2.fromJson(modelJson);
  //   const predY = await lr2.predict(cases);
  //   const acc = accuracyScore(yMult, predY);
  //   console.log('accuracy:', acc);
  //   assert.isTrue(acc >= 0.75);
  // });
  // it('save and load model as predictor', async () => {
  //   const lr = new LogisticRegression();
  //   await lr.fit(cases, y);
  //   const modelJson = await lr.toJson();
  //   const lr2 = new LogisticRegressionPredictor();
  //   await lr2.fromJson(modelJson);
  //   const predY = await lr2.predict(cases);
  //   const acc = accuracyScore(y, predY);
  //   console.log('accuracy:', acc);
  //   assert.isTrue(acc >= 0.95);
  // });
  // it('save and load model as predictor (multi-classification)', async () => {
  //   const lr = new LogisticRegression();
  //   await lr.fit(cases, yMult);
  //   const modelJson = await lr.toJson();
  //   const lr2 = new LogisticRegressionPredictor();
  //   await lr2.fromJson(modelJson);
  //   const predY = await lr2.predict(cases);
  //   const acc = accuracyScore(yMult, predY);
  //   console.log('accuracy:', acc);
  //   assert.isTrue(acc >= 0.75);
  // });
});
