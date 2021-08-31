import { eigenSolve } from '../../../src/linalg/eigen';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { assert } from 'chai';
import 'mocha';
import { max } from '@tensorflow/tfjs-core';

const matrix = tf.tensor2d([
  [ 1, 2, 5, 10 ],
  [ 2, 6, 7, 5 ],
  [ 5, 7, 9, 6 ],
  [ 10, 5, 6, 7]
]);

describe('EigenSolver', () => {
    
  it('spectrum decomposition', () => {
    const [ d, q ] = eigenSolve(matrix);
    const di = tf.diag(d);
    const recovM = tf.matMul(tf.matMul(tf.transpose(q), di), q);
    const dm = tf.matMul(tf.matMul(q, matrix), tf.transpose(q));
    const equal = tensorEqual(recovM, matrix, 1e-3);
    assert.isTrue(equal);
  })
});
