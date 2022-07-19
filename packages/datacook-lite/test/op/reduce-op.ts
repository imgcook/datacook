// import { add2d, div2d, mul2d } from "../../src/core/op/binary-op";
import { matrix, Scalar, vector } from '../../src/core/classes';
import { assert } from 'chai';
import { matMul2d, sum2d, mean2d, min2d } from "../../src/core/op";
import { max1d, max2d, mean1d, min1d, mul1d, sum1d } from '../../src/backend-cpu/op';
import { Vector } from '../../src/backend-cpu/classes';
import { ByAxis } from '../../src/backend-cpu/op/basic-impl';

describe('Reduce OP Test', () => {
  it('sum2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = sum2d(a) as Vector;
    const d = mul1d(c, 2);
    d.backward();
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });
  it('sum2d reduce all', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = sum2d(a, -1);
    c.backward();
    console.log('c', c.data);
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });
  it('mean2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = mean2d(a) as Vector;
    c.backward();
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });
  it('mean2d reduce all', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = mean2d(a, -1) as Vector;
    c.backward();
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });
  it('min2d by column', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ], [ 0, 5 ] ]);
    const c = min2d(a, ByAxis.ByColumn) as Vector;
    c.backward();
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });
  it('min2d by row', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ], [ 0, 5 ] ]);
    const c = min2d(a, ByAxis.ByRow) as Vector;
    c.backward();
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });
  it('min2d reduce all', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ], [ 0, 5 ] ]);
    const c = min2d(a, -1) as Scalar;
    c.backward();
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });

  it('max2d by column', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ], [ 0, 5 ] ]);
    const c = max2d(a, ByAxis.ByColumn) as Vector;
    c.backward();
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });
  it('max2d by row', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ], [ 0, 5 ] ]);
    const c = max2d(a, ByAxis.ByRow) as Vector;
    c.backward();
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });
  it('max2d reduce all', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ], [ 0, 5 ] ]);
    const c = max2d(a, -1) as Scalar;
    c.backward();
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });

  it('sum1d', () => {
    const a = vector([ 1, 2, 3, 4, 5 ]);
    const b = sum1d(a);
    b.backward();
    console.log('grad a', a.grad.data);
  });
  it('mean1d', () => {
    const a = vector([ 1, 2, 3, 4, 5 ]);
    const b = mean1d(a);
    b.backward();
    console.log('grad a', a.grad.data);
  });
  it('min1d', () => {
    const a = vector([ 1, 2, 3, 4, 5 ]);
    const b = min1d(a);
    b.backward();
    console.log('grad a', a.grad.data);
  });
  it('max1d', () => {
    const a = vector([ 1, 2, 3, 4, 5 ]);
    const b = max1d(a);
    b.backward();
    console.log('grad a', a.grad.data);
  });
});
