import { getQuantile } from '../../../src/stat/data';
import { tensor } from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import { tensorEqual } from '../../../src/linalg';
import { quickSelectQuantiles } from '../../../src/stat/utils';

const x = [
  [ 1, 2, 3, 4, 5, 6 ],
  [ 4, 5, 6, 7, 8, 9 ],
  [ 7, 8, 9, 10, 11, 12 ],
  [ 4, 5, 6, 1, 2, 3 ]
];


describe('Calculate Quantile', () => {
  it('quick select quantile', () => {
    const arr = [ 2, 4, 2, 8, 10, 1 ];
    const quantiles = quickSelectQuantiles(arr, [0, 0.25, 0.5, 0.75, 1]);
    assert.isTrue(tensorEqual(tensor(quantiles), tensor([ 1, 2, 3, 7, 10 ])));
    const arr2 = [ 2, 4, 2, 8, 10, 1, 5 ];
    const quantiles2 = quickSelectQuantiles(arr2, [0, 0.25, 0.5, 0.75, 1]);
    assert.isTrue(tensorEqual(tensor(quantiles2), tensor([ 1, 2, 4, 6.5, 10 ])));
  });
  it('get quantile (axis = -1)', () => {
    const  quantiles = getQuantile(x, [0, 0.25, 0.5, 0.75, 1]);
    assert.isTrue(tensorEqual(quantiles, tensor([ 1, 3.75, 5.5, 8, 12 ])));
  });
  it('get quantile (axis = 1)', () => {
    const  quantiles = getQuantile(x, [0, 0.25, 0.5, 0.75, 1], 1);
    assert.isTrue(tensorEqual(quantiles, tensor(
        [[1, 2.25, 3.5, 4.75 , 6 ],
        [4, 5.25, 6.5, 7.75 , 9 ],
        [7, 8.25, 9.5, 10.75, 12],
        [1, 2.25, 3.5, 4.75 , 6 ]]
    )));
  });
  it('get quantile (axis = 0)', () => {
    const  quantiles = getQuantile(x, [0, 0.25, 0.5, 0.75, 1], 0);
    quantiles.print();
    assert.isTrue(tensorEqual(quantiles, tensor(
        [[1, 3.25, 4  , 4.75, 7 ],
        [2, 4.25, 5  , 5.75, 8 ],
        [3, 5.25, 6  , 6.75, 9 ],
        [1, 3.25, 5.5, 7.75, 10],
        [2, 4.25, 6.5, 8.75, 11],
        [3, 5.25, 7.5, 9.75, 12]]
    )));
  });
  it('get quantile (1d array)', () => {
    const arr = [ 2, 4, 2, 8, 10, 1 ];
    const quantiles = getQuantile(arr, [0, 0.25, 0.5, 0.75, 1]);
    assert.isTrue(tensorEqual(quantiles, tensor([1, 2, 3, 7, 10])));
  });
});