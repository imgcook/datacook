import { LinearRegression } from '../../../src/model/linear-model/linear-regression';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import 'mocha';
import { Tensor } from '@tensorflow/tfjs-core';

const cases = tf.mul(tf.randomNormal([200, 5]),[ 1, 10, 100, 2, 3 ]);
const weight = tf.tensor([ 2, 3, 1, 4, 6 ])
const y = tf.add(tf.sum(tf.mul(cases, weight), 1), 10);

describe('Linear Regression', () => {
    it('train simple dataset', async () => {
        const lm = new LinearRegression({});
        await lm.train(cases, y); 
        const { coefficients, intercept } = lm.getCoef();
        coefficients.print();
        intercept.print();
        const predY = lm.predict(cases, y);
        /*if (predY instanceof Tensor){
            predY.print();
        }*/
    })
})