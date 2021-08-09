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
const labels = [0, 0, 0, 5];

const testCases = tf.tensor2d([
    [2, 0, 0, 0, 0, 0],
    [6, 0, 0, 0, 0, 0],
    [1, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 4, 6]
  ]);
const testLabels = [0, 0, 0, 5];

describe('Naive bayes', () => {
    it('predict train case', async () => {
        const mnb = new MultinomialNB();
        // train twice
        await mnb.train(cases, labels);
        await mnb.train(cases, labels);
        const prediction = mnb.predict(cases);
        prediction.print();
        assert.deepEqual(prediction.arraySync(), labels);
    });
    it('predict class probabilities', async () => {
        const Xtest = tf.tensor2d([[2, 1, 0, 0, 0, 0],[3, 0, 0, 0, 1, 1],[1,0,0,0,1,1]]);
        const expection = tf.tensor1d([0]);
        const mnb = new MultinomialNB();
        await mnb.train(cases, labels);
        const probs = mnb.predictProba(cases);
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
        assert.deepEqual(yPred.arraySync(), testLabels);
    });

    it('save and load model', async () => {
        const mnb = new MultinomialNB();
        await mnb.train(cases, labels);
        const modelJson = mnb.toJson();
        const mnb2 = new MultinomialNB();
        mnb2.load(modelJson);
        const yPred = mnb2.predict(testCases);
        assert.deepEqual(yPred.arraySync(), testLabels);
    });
});