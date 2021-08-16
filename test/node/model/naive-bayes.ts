
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
    const prediction = mnb.predict(cases);
    prediction.print();
    // @ts-ignore
    assert.deepEqual(prediction.arraySync(), labels);
  });

  it('predict class probabilities', async () => {
    const mnb = new MultinomialNB();
    await mnb.train(cases, labels);
    const probs = mnb.predictProba(cases);
    // probs.print();
    const nCases = cases.shape[0];
    const class_prob_sum = tf.sum(probs, 1);
    const class_sum = new Array(nCases).fill(1);
    assert.deepEqual(class_prob_sum.arraySync(), class_sum);
  });

  it('predict test case', async () => {
    const mnb = new MultinomialNB();
    await mnb.train(cases, labels);
    const yPred = mnb.predict(testCases);
    yPred.print();
    // @ts-ignore
    assert.deepEqual(yPred.arraySync(), testLabels);
  });

  it('save and load model', async () => {
    const mnb = new MultinomialNB();
    await mnb.train(cases, labels);
    const modelJson = mnb.toJson();
    const mnb2 = new MultinomialNB();
    mnb2.load(modelJson);
    const yPred = mnb2.predict(testCases);
    // @ts-ignore
    assert.deepEqual(yPred.arraySync(), testLabels);
  });

});
