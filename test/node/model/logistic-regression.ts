import { LogisticRegression } from '../../../src/model/linear-model/logistic-regression';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import 'mocha';
import { Tensor } from '@tensorflow/tfjs-core';
import { accuracyScore } from '../../../src/metrics/classifier';

const cases = tf.mul(tf.randomNormal([10000, 5]),[ 10, 10, 10, 20, 10 ]);
const weight = tf.tensor([ 2, 3, 1, 4, 6 ])
const eta = tf.add(tf.sum(tf.mul(cases, weight), 1), -20);
const y = tf.greater(tf.sigmoid(eta), 0.5);

describe('Logistic Regression', () => {
	it('train simple dataset', async () => {
		const lr = new LogisticRegression({fitIntercept: true});
		await lr.train(cases, y); 
		const { coefficients, intercept } = lr.getCoef();
		const predY = lr.predict(cases);
		if (predY instanceof Tensor){
			y.print();
			predY.print();
			const acc = accuracyScore(y, predY);
			console.log(acc);
			const coefs = lr.getCoef();
			coefs.coefficients.print();
			coefs.intercept.print();
		}
	});
	it('train simple dataset on batch', async () => {
		const lr = new LogisticRegression();	
		for (let i = 0; i < 800; i++) {
			const j = Math.floor(i%(100));
			const batchX = tf.slice(cases, [j * 100, 0], [100 ,5]);
			const batchY = tf.slice(y, [j * 100], [100]);
			await lr.trainOnBatch(batchX, batchY, 0.01)
		}
		const predY = lr.predict(cases);
		if (predY instanceof Tensor){
			y.print();
			predY.print();
			const acc = accuracyScore(y, predY);
			console.log(acc);
			const coefs = lr.getCoef();
			coefs.coefficients.print();
			coefs.intercept.print();
		}

	})
	
})