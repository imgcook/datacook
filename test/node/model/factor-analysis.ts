import { FactorAnalysis } from '../../../src/model/factor-analysis/index';
import * as tf from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg';
import { assert } from 'chai';

const matrix = tf.tensor2d([
  [ 1, 5, 7, 6, 1 ],
  [ 2, 1, 10, 4, 4 ],
  [ 3, 6, 7, 5, 2 ],
  [ 6, 7, 8, 9, 10 ],
  [ 8, 8, 9, 9, 5 ],
  [ 10, 2, 4, 5, 2 ]
]);

// result computed in sklearn
const componentsExp = tf.tensor2d([
  [ 1.34563819e+00, 2.80961604e+00, 2.12909870e-01 ],
  [ 2.17661384e+00, -7.44052185e-01, 8.43185285e-01 ],
  [ 3.89695778e-01, -1.12233215e+00, -8.50008393e-01 ],
  [ 1.92648813e+00, -3.68052325e-02, -7.37909482e-03 ],
  [ 2.15692479e+00, -1.21229029e-03, -1.81701396e+00 ]
]);

describe('Factor analysis', () => {
  it('fit factor loadings', async () => {
    const fa = new FactorAnalysis({nComponent: 3});
    await fa.fit(matrix);
    fa.factorLoadings.print();
    assert.isTrue(tensorEqual(fa.factorLoadings, componentsExp, 1e-1));
  });
});