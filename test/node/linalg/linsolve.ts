import '@tensorflow/tfjs-backend-cpu';
import { tensor, matMul,reshape } from '@tensorflow/tfjs-core';
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
  
  it('solve matrix', () => {
    const solve = linSolveQR(matrix, v);
    solve.print();
    matMul(matrix, reshape(solve, [-1, 1])).print();
    
  })
})