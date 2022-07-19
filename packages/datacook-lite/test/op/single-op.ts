import { matrix, Scalar, vector } from '../../src/core/classes';
import { assert } from 'chai';
import { sum2d, mean2d, min2d, exp2d, abs2d, pow2d, sigmoid2d } from "../../src/core/op";
import { Vector } from '../../src/backend-cpu/classes';
import { ByAxis } from '../../src/backend-cpu/op/basic-impl';
import { neg2d, sqrt2d, square2d } from '../../src/backend-cpu/op';
import { log2d } from '../../src/backend-cpu/op/single-op';

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
  it('square2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = square2d(a);
    c.backward();
    console.log('grad a', a.grad.data);
  });
  it('log2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = log2d(a);
    c.backward();
    console.log('grad a', a.grad.data);
  });
  it('exp2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = exp2d(a);
    c.backward();
    console.log('grad a', a.grad.data);
  });
  it('abs2d', () => {
    const a = matrix([ [ 1, 0 ], [ -3, 0 ] ]);
    const c = abs2d(a);
    c.backward();
    console.log('grad a', a.grad.data);
  });
  it('pow2d', () => {
    const a = matrix([ [ 1, 0 ], [ -3, 0 ] ]);
    const c = pow2d(a, 3);
    c.backward();
    console.log('grad a', a.grad.data);
  });
  it('sigmoid2d', () => {
    const a = matrix([ [ 1, 0 ], [ -3, 0 ] ]);
    const c = sigmoid2d(a);
    c.backward();
    console.log('grad a', a.grad.data);
  });

});
