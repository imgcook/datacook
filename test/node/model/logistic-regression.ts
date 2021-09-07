import { LogisticRegression } from '../../../src/model/linear-model/logistic-regression';
import { LogisticRegressionPredictor } from '../../../src/model/linear-model/logistic-regression-predictor';
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

describe('Logistic ', () => {

	it('predict class probabilities', async () => {
    const lr = new LogisticRegression({fitIntercept: true});
		await lr.train(cases, y); 
    const probs = await lr.predictProba(cases);
    const trueCount = tf.sum(tf.lessEqual(probs, 1)).dataSync()[0];
		assert.deepEqual(trueCount, 10000);
  });

	it('train simple dataset', async () => {
		const lr = new LogisticRegression({fitIntercept: true});
		await lr.train(cases, y); 
		const predY = await lr.predict(cases);
		if (predY instanceof Tensor){
			const acc = accuracyScore(y, predY);
			assert.isTrue(acc >= 0.95);
		}
	});

	it('train simple dataset on batch', async () => {
		const lr = new LogisticRegression({optimizerType: 'adam', optimizerProps: {learningRate: 0.01}});	
		for (let i = 0; i < 800; i++) {
			const j = Math.floor(i%(100));
			const batchX = tf.slice(cases, [j * 100, 0], [100 ,5]);
			const batchY = tf.slice(y, [j * 100], [100]);
			await lr.trainOnBatch(batchX, batchY)
		}
		const predY = await lr.predict(cases);
		if (predY instanceof Tensor){
			const acc = accuracyScore(y, predY);
			assert.isTrue(acc >= 0.95);
		}
	});
	it('save and load model', async () => {
		const lr = new LogisticRegression({fitIntercept: true});
		await lr.train(cases, y); 
		const modelJson = await lr.toJson();
		const lr2 = new LogisticRegression();
		await lr2.fromJson(modelJson);
		const predY = await lr2.predict(cases);
		if (predY instanceof Tensor){
			const acc = accuracyScore(y, predY);
			assert.isTrue(acc >= 0.95);
		}
	});
	it('save and load model as predictor', async () => {
		const lr = new LogisticRegression({fitIntercept: true});
		await lr.train(cases, y); 
		const modelJson = await lr.toJson();
		const lr2 = new LogisticRegressionPredictor();
		await lr2.fromJson(modelJson);
		const predY = await lr2.predict(cases);
		if (predY instanceof Tensor){
			const acc = accuracyScore(y, predY);
			assert.isTrue(acc >= 0.95);
		}
	});
});
