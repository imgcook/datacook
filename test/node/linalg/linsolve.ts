import '@tensorflow/tfjs-backend-cpu';
<<<<<<< HEAD
import { tensor, matMul, reshape, sub, squeeze } from '@tensorflow/tfjs-core';
=======
import { tensor, matMul, reshape, sub } from '@tensorflow/tfjs-core';
>>>>>>> 3656f635f8e7ba2b74385608a932249c54c1bed4
import { tensorEqual } from '../../../src/linalg/utils';
import { linSolveQR } from '../../../src/linalg/linsolve';
import { assert } from 'chai';
import 'mocha';
const matrix = tensor([
    [1, -2, 1],
    [0, 2, -8],
    [5, 0, -5]
])
const v = tensor([0, 8, 10]);

describe('linSolver', () => {
<<<<<<< HEAD
  it('solve matrix', () => {
    const solve = linSolveQR(matrix, v);
    const predv = squeeze(matMul(matrix, reshape(solve, [-1, 1])));
=======
  
  it('solve matrix', () => {
    const solve = linSolveQR(matrix, v);
    const predv = matMul(matrix, reshape(solve, [-1, 1]));
>>>>>>> 3656f635f8e7ba2b74385608a932249c54c1bed4
    const equal = tensorEqual(v, predv, 1e-3);
    assert.isTrue(equal);
  })
})