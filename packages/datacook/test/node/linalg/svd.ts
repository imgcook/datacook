import { svd } from '../../../src/linalg/svd';
import * as tf from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { assert } from 'chai';
import 'mocha';

const matrix = tf.tensor2d([
  [ 1, 5, 7, 6, 1 ],
  [ 2, 1, 10, 4, 4 ],
  [ 3, 6, 7, 5, 2 ]
]);
const singularValues = tf.tensor([ 18.54, 5.01, 1.83 ]);

describe('SVDSolver', () => {
  it('svd decomposition', async () => {
    const d = (await svd(matrix))[1];
    const equal = tensorEqual(d, singularValues, 1e-2);
    assert.isTrue(equal);
  });
});

