import { getCorrelationMatrix, getCovarianceMatrix } from '../../../src/stat/corcov';
import { tensor } from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import { tensorEqual } from '../../../src/linalg';

const m = [
  [ 12, 43, 23, 23 ],
  [ 15, 34, 24, 34 ],
  [ 16, 53, 13, 64 ],
  [ 31, 64, 21, 53 ],
  [ 25, 51, 13, 61 ]
];

// correlation result calculated in numpy
const corrMAssert = tensor([
  [ 1., 0.79160783, -0.24071992, 0.57677011 ],
  [ 0.79160783, 1., -0.43190936, 0.65594465 ],
  [ -0.24071992, -0.43190936, 1., -0.87117726 ],
  [ 0.57677011, 0.65594465, -0.87117726, 1. ]
]);

// covariance result calculated in numpy
const covMAssert = tensor([
  [ 62.7, 70.5, -10.3 , 81.25 ],
  [ 70.5, 126.5, -26.25, 131.25 ],
  [ -10.3, -26.25, 29.2, -83.75 ],
  [ 81.25, 131.25, -83.75, 316.5 ]
]);

describe('Covariance and correlation', () => {
  it('get correlation', () => {
    const corrM = getCorrelationMatrix(m);
    assert(tensorEqual(corrM, corrMAssert, 1e-4));
  });

  it('get covariance', () => {
    const covM = getCovarianceMatrix(m);
    assert(tensorEqual(covM, covMAssert, 1e-4));
  });
});
