
import { MultinomialNB } from '@pipcook/datacook/dist/model/naive-bayes';
import { MultinomialNBPredictor } from '../../src/model/naive-bayes/naive-bayes-predictor';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import 'mocha';

const cases = [
  [ 2, 1, 0, 0, 0, 0 ],
  [ 2, 0, 1, 0, 0, 0 ],
  [ 1, 0, 0, 1, 0, 0 ],
  [ 1, 0, 0, 0, 1, 1 ]
];
const labels = [ 'A', 'A', 'A', 'B' ];

const testCases = [
  [ 2, 0, 0, 0, 0, 0 ],
  [ 6, 0, 0, 0, 0, 0 ],
  [ 1, 0, 0, 3, 0, 0 ],
  [ 0, 0, 0, 0, 4, 6 ]
];
const testLabels = [ 'A', 'A', 'A', 'B' ];

describe('Naive bayes', () => {

  it('predict train case', async () => {
    const mnb = new MultinomialNB();
    // train twice
    await mnb.train(cases, labels);
    await mnb.train(cases, labels);
    const modelJson = mnb.toJson();
    const mnb2 = new MultinomialNBPredictor();
    await mnb2.load(modelJson);
    const prediction = await mnb2.predict(cases);
    assert.deepEqual(prediction as any, labels);
  });

  it('predict test case', async () => {
    const mnb = new MultinomialNB();
    // train twice
    await mnb.train(cases, labels);
    await mnb.train(cases, labels);
    const modelJson = mnb.toJson();
    const mnb2 = new MultinomialNBPredictor();
    await mnb2.load(modelJson);
    const prediction = await mnb2.predict(testCases);
    assert.deepEqual(prediction as any, testLabels);
  });

  it('predict class probabilities', async () => {
    const mnb = new MultinomialNB();
    await mnb.train(cases, labels);
    const modelJson = mnb.toJson();
    const mnb2 = new MultinomialNBPredictor();
    await mnb2.load(modelJson);
    const probs = await mnb2.predictProba(cases);
    const class_prob_sum = tf.sum(probs, 1);
    const class_sum = new Array(cases.length).fill(1);
    assert.deepEqual(class_prob_sum.arraySync(), class_sum);
  });

});
