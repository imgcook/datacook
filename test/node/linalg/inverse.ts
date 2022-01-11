import { inverse } from '../../../src/linalg/inverse';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { assert } from 'chai';
import 'mocha';
const matrix = tf.tensor2d([
  [ 1, 2, 5, 10 ],
  [ 2, 6, 7, 5 ],
  [ 5, 7, 9, 6 ],
  [ 10, 5, 6, 7 ]
]);

const singularMatrix = tf.tensor2d([
  [ 1, 1, 3, 4 ],
  [ 2, 2, 3, 4 ],
  [ 3, 3, 6, 7 ],
  [ 4, 4, 7, 8 ]
]);

describe('Matrix Solver', () => {

  it('get inverse matrix', async () => {
    const invM = await inverse(matrix);
    const iM = tf.matMul(matrix, invM);
    const isIdMatrix = tensorEqual(iM, tf.eye(iM.shape[0]), 1e-2);
    assert.isTrue(isIdMatrix);
  });

  it('solve singular matrix throw type error', async () => {
    let err: Error;
    try {
      const invM = await inverse(singularMatrix);
    } catch (error) {
      err = error;
    }
    assert.isTrue(err && err instanceof TypeError);
  });

});
