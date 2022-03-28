import { getVariance } from '../../../src/stat/data';
import { numEqual } from '../../../src/math/utils';
import { tensor } from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import { tensorEqual } from '../../../src/linalg';
const x = [
  [
    [
      [ 1, 2, 3 ],
      [ 4, 5, 6 ],
      [ 7, 8, 9 ],
      [ 4, 5, 6 ] ],
    [
      [ 1, 2, 3 ],
      [ 4, 5, 6 ],
      [ 8, 8, 9 ],
      [ 4, 5, 9 ]
    ]
  ],
  [
    [
      [ 1, 2, 4 ],
      [ 4, 5, 6 ],
      [ 7, 8, 9 ],
      [ 4, 5, 6 ] ],
    [ [ 1, 2, 3 ],
      [ 4, 5, 6 ],
      [ 7, 8, 9 ],
      [ 4, 5, 6 ]
    ]
  ]
];
describe('Covariance and correlation', () => {
  it('get variance (axis = -1)', () => {
    const variance = getVariance(x).dataSync()[0];
    const varianceTrue = 5.627216312056737;
    assert.isTrue(numEqual(variance, varianceTrue, 1e-4));
  });
  it('get variance (axis = 0)', () => {
    const variance = getVariance(x, 0);
    const varianceTrue =
    [ [ [ 0., 0., 0.5 ],
      [ 0., 0., 0. ],
      [ 0., 0., 0. ],
      [ 0., 0., 0. ] ],
    [ [ 0., 0., 0. ],
      [ 0., 0., 0. ],
      [ 0.5, 0., 0. ],
      [ 0., 0., 4.5 ] ] ];
    assert.isTrue(tensorEqual(variance, tensor(varianceTrue), 1e-4));
  });
  it('get variance (axis = 2)', () => {
    const variance = getVariance(x, 2);
    const varianceTrue = [
      [
        [ 6., 6., 6. ],
        [ 8.25, 6., 8.25 ]
      ],

      [
        [ 6., 6., 4.25 ],
        [ 6., 6., 6. ]
      ]
    ];
    assert.isTrue(tensorEqual(variance, tensor(varianceTrue), 1e-4));
  });
});
