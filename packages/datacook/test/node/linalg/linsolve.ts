import { tensor, matMul, reshape, squeeze } from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { linSolveQR } from '../../../src/linalg/linsolve';
import { assert } from 'chai';
import 'mocha';
const matrix = tensor([
  [ 1, -2, 1 ],
  [ 0, 2, -8 ],
  [ 5, 0, -5 ]
]);
const v = tensor([ 0, 8, 10 ]);

describe('linSolver', () => {
  it('solve matrix', async () => {
    const solve = await linSolveQR(matrix, v);
    const predv = squeeze(matMul(matrix, reshape(solve, [ -1, 1 ])));
    const equal = tensorEqual(v, predv, 1e-3);
    assert.isTrue(equal);
  });
});
