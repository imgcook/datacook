
import { BaseEstimator } from '../../../src/model/base';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';

class TestEstimator extends BaseEstimator {
  public a: tf.Tensor;
  constructor(a: tf.Tensor) {
    super();
    this.a = a;
  }
}

describe('Base Model', () => {
  it('clean model tensor', () => {
    const nBaseTensors = tf.memory().numTensors;
    const a = tf.tensor([ 0, 1, 2 ]);
    const testEstimator = new TestEstimator(a);
    testEstimator.clean();
    assert.isTrue((tf.memory().numTensors - nBaseTensors) === 0);
    
  });
  it('reset model tensor parameter', () => {
    const nBaseTensors = tf.memory().numTensors;
    const a = tf.tensor([ 0, 1, 2 ]);
    const b = tf.tensor([ 1, 2, 3 ]);
    const testEstimator = new TestEstimator(a);
    testEstimator.cleanAndSetParam('a', b);
    assert.isTrue((tf.memory().numTensors - nBaseTensors) === 1);
    testEstimator.clean();
    assert.isTrue((tf.memory().numTensors - nBaseTensors) === 0);
  });
});
