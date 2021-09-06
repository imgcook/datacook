import { LinearRegression } from '../../../src/model/linear-model/linear-regression';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import { tensorEqual } from '../../../src/linalg/utils';
import 'mocha';
import { Tensor } from '@tensorflow/tfjs-core';

const cases = tf.mul(tf.randomNormal([10000, 5]),[ 1, 10, 100, 2, 3 ]);
const weight = tf.tensor([ 2, 3, 1, 4, 6 ])
const y = tf.add(tf.sum(tf.mul(cases, weight), 1), 10);

describe('Linear Regression', () => {
    it('train simple dataset', async () => {
        const optimizer = tf.train.adam(0.5);
        const lm = new LinearRegression({optimizer});
        await lm.train(cases, y); 
        const { coefficients } = lm.getCoef();
        const equal = tensorEqual(coefficients, weight, 1e-2);
        assert.isTrue(equal);
    })
    it('train simple dataset on batch', async () => {
		const optimizer = tf.train.adam(0.1);
		const lm = new LinearRegression({optimizer});	
		for (let i = 0; i < 800; i++) {
			const j = Math.floor(i%(100));
			const batchX = tf.slice(cases, [j * 100, 0], [100 ,5]);
			const batchY = tf.slice(y, [j * 100], [100]);
			await lm.trainOnBatch(batchX, batchY)
		}
        const { coefficients } = lm.getCoef();
		const equal = tensorEqual(coefficients, weight, 1e-2);
        assert.isTrue(equal);
	})
})