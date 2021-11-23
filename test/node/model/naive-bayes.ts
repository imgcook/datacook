
import { MultinomialNB } from '../../../src/model/naive-bayes';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import 'mocha';

const cases = tf.tensor2d([
  [2, 1, 0, 0, 0, 0],
  [2, 0, 1, 0, 0, 0],
  [1, 0, 0, 1, 0, 0],
  [1, 0, 0, 0, 1, 1]
]);
const labels = ['A', 'A', 'A', 'B'];

const testCases = tf.tensor2d([
  [2, 0, 0, 0, 0, 0],
  [6, 0, 0, 0, 0, 0],
  [1, 0, 0, 3, 0, 0],
  [0, 0, 0, 0, 4, 6]
]);
const testLabels = ['A', 'A', 'A', 'B'];

describe('Naive bayes', () => {

  it('predict train case', async () => {
    const mnb = new MultinomialNB();
    // train twice
    await mnb.train(cases, labels);
    await mnb.train(cases, labels);
<<<<<<< HEAD
    const prediction = mnb.predict(cases);
    // @ts-ignore
    assert.deepEqual(prediction.arraySync(), labels);
=======
    const prediction = await mnb.predict(cases);
    assert.deepEqual(prediction.arraySync() as any, labels);
>>>>>>> a204a8a299a83931037e6addd72fe7816251b3e4
  });

  it('predict class probabilities', async () => {
    const mnb = new MultinomialNB();
    await mnb.train(cases, labels);
    const probs = await mnb.predictProba(cases);
    const nCases = cases.shape[0];
    const class_prob_sum = tf.sum(probs, 1);
    const class_sum = new Array(nCases).fill(1);
    assert.deepEqual(class_prob_sum.arraySync(), class_sum);
  });

  it('predict test case', async () => {
    const mnb = new MultinomialNB();
    await mnb.train(cases, labels);
<<<<<<< HEAD
    const yPred = mnb.predict(testCases);
    // @ts-ignore
    assert.deepEqual(yPred.arraySync(), testLabels);
=======
    const yPred = await mnb.predict(testCases);
    assert.deepEqual(yPred.arraySync() as any, testLabels);
>>>>>>> a204a8a299a83931037e6addd72fe7816251b3e4
  });

  it('save and load model', async () => {
    const mnb = new MultinomialNB();
    await mnb.train(cases, labels);
    const modelJson = mnb.toJson();
    const mnb2 = new MultinomialNB();
<<<<<<< HEAD
    mnb2.load(modelJson);
    const yPred = mnb2.predict(testCases);
    // @ts-ignore
    assert.deepEqual(yPred.arraySync(), testLabels);
=======
    await mnb2.load(modelJson);
    const yPred = await mnb2.predict(testCases);
    assert.deepEqual(yPred.arraySync() as any, testLabels);
>>>>>>> a204a8a299a83931037e6addd72fe7816251b3e4
  });

});
