import { MLPRegressor } from '../../../src/model/neural-network/MLPRegressor';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { getRSquare } from '../../../src/metrics/regression';
import { assert } from 'chai';
import 'mocha';
import { Tensor1D } from '@tensorflow/tfjs-core';

const cases = tf.mul(tf.randomNormal([ 10000, 5 ]), [ 1, 10, 100, 2, 3 ]);
const weight = tf.tensor([ 2, 3, 1, 4, 6 ]);
const y = tf.add(tf.sum(tf.mul(cases, weight), 1), 10);

describe('MLP Regression', () => {

  it('train simple dataset', async () => {
    const mlp = new MLPRegressor({
      hiddenLayerSizes: [ 4, 10 ],
      activations: 'relu',
      optimizerType: 'adam',
      optimizerProps: { learningRate: 0.01 }
    });
    await mlp.fit(cases, y);
    const yPred = await mlp.predict(cases) as Tensor1D;
    assert.isTrue(getRSquare(yPred, y as Tensor1D) > 0.8);
  });

  it('save and load model', async () => {
    const mlp = new MLPRegressor({
      hiddenLayerSizes: [ 4, 10 ],
      activations: 'relu',
      optimizerType: 'adam',
      optimizerProps: { learningRate: 0.01 }
    });
    await mlp.fit(cases, y, { verbose: true });
    const modelJson = await mlp.toJson();
    const mlp2 = new MLPRegressor({});
    mlp2.fromJson(modelJson);
    const yPred = await mlp2.predict(cases);
    assert.isTrue(getRSquare(yPred as Tensor1D, y as Tensor1D) > 0.8);
  });
});
