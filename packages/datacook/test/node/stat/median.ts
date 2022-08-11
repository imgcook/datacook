import { getMedian } from '../../../src/stat/data';
import { numEqual } from '../../../src/math/utils';
import { tensor, Scalar, scalar } from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import { tensorEqual } from '../../../src/linalg';
import { quickSelect, randomChoice, quickSelectMedian } from '../../../src/stat/utils';

const x = [
  [ 1, 2, 3 ],
  [ 4, 5, 6 ],
  [ 7, 8, 9 ],
  [ 4, 5, 6 ]
];


describe('Calculate Median', () => {
  it ('quick select', () => {
      const arr = [ 2, 4, 2, 8, 10, 1 ];
      const d0 = quickSelect(arr, 0, randomChoice);
      const d1 = quickSelect(arr, 1, randomChoice);
      const d2 = quickSelect(arr, 2, randomChoice);
      const d3 = quickSelect(arr, 3, randomChoice);
      const d4 = quickSelect(arr, 4, randomChoice);
      assert.isTrue(d0 === 1);
      assert.isTrue(d1 === 2);
      assert.isTrue(d2 === 2);
      assert.isTrue(d3 === 4);
      assert.isTrue(d4 === 8);
  });
  it('quick select median', () => {
    const arr = [ 2, 4, 2, 8, 10, 1 ];
    const median = quickSelectMedian(arr);
    assert.isTrue(median === 3);
  });
  it('get median (axis = -1)', () => {
    const median = getMedian(x);
    assert.isTrue(tensorEqual(median, scalar(5)));
  });
  it('get median (axis = 1)', () => {
    const median = getMedian(x, 1);
    assert.isTrue(tensorEqual(median, tensor([ 2, 5, 8, 5 ])));
  });
  it('get median (axis = 0)', () => {
    const median = getMedian(x, 0);
    assert.isTrue(tensorEqual(median, tensor([ 4, 5, 6 ])));
  });
  it('get median (1d array)', () => {
    const median = getMedian([1, 4, 7, 2, 4, 6, 0, 8]);
    assert.isTrue(tensorEqual(median, scalar(4)));
  });
});