import { eigenSolve } from '../../../src/linalg/eigen';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { assert } from 'chai';
import 'mocha';

const matrix = tf.tensor2d([
  [ 1, 2, 5, 10 ],
  [ 2, 6, 7, 5 ],
  [ 5, 7, 9, 6 ],
  [ 10, 5, 6, 7]
]);

describe('EigenSolver', () => {
    
  it('spectrum decomposition', async () => {
    const [ d, q ] = await eigenSolve(matrix);
    const di = tf.diag(d);
    const recovM = tf.matMul(tf.matMul(q, di), tf.transpose(q));
    const equal = tensorEqual(recovM, matrix, 1e-3);
    assert.isTrue(equal);
  });
});
