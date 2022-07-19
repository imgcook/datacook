import { matrix, Scalar, vector } from '../../src/core/classes';
import { assert } from 'chai';
import { sum2d, mean2d, min2d } from "../../src/core/op";
import { Vector } from '../../src/backend-cpu/classes';
import { ByAxis } from '../../src/backend-cpu/op/basic-impl';
import { neg2d, sqrt2d } from '../../src/backend-cpu/op';

describe('Single OP Test', () => {
  it('neg2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = neg2d(a);
    c.backward();
    console.log('grad a', a.grad.data);
  });
  it('sqrt2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = sqrt2d(a);
    c.backward();
    console.log('grad a', a.grad.data);
  });
});
