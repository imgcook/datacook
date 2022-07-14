import { add2d, div2d, mul2d } from "../../src/core/op/binary-op";
import { matrix } from '../../src/core/classes';
import { assert } from 'chai';
import { matMul2d } from "../../src/core/op";
import { scalar, vector } from "../../src/core/classes/creation";

describe('OP Test', () => {
  it('add2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = add2d(a, 2);
    assert(c.get(0, 0) === 3);
  });
  it('matmul2d', () => {
    const a = matrix([ [ 2, 3 ], [ 4, 5 ] ]);
    const b = matrix(([ [ -2, -3 ], [ -3, -5 ] ]));
    const c = matMul2d(a, b);
    c.backward();
    console.log(c.grad.data);
    console.log(a.grad.data);
    console.log(b.grad.data);
  });
  it('add2d and matmul', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = scalar(2);
    const c = add2d(a, b);
    const d = matrix(([ [ -2, -3 ], [ -3, -5 ] ]));
    const e = matMul2d(c, d);
    e.backward();
    console.log('grad d', d.grad.data);
    console.log('grad c', c.grad.data);
    console.log('grad b', b.grad.data);
    console.log('grad a', a.grad.data);
  });

  it('mul', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = scalar(2);
    const c = mul2d(a, b);
    const gradA = [ [ 2, 2 ], [ 2, 2 ] ];
    c.backward();
    console.log('grad c', c.grad.data);
    console.log('grad b', b.grad.data);
    console.log('grad a', a.grad.data);
    assert.isTrue(b.grad.data === 10);
    // assert.isTrue(a.grad === gradA);
  });
});

describe('OP div', () => {
  it('div2d matrix', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = matrix([ [ 1, 2 ], [ 1, 2 ] ]);
    const c = div2d(a, b);
    const gradA = [ [ 2, 2 ], [ 2, 2 ] ];
    c.backward();
    console.log('grad a', a.grad.data);
  });
  it('div2d vector', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = vector([ 1, 2 ]);
    const c = div2d(a, b);
    c.backward();
    console.log('grad b', b.grad.data);
  });
  it('div2d scalar', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = scalar(2);
    const c = div2d(a, b);
    const gradA = [ [ 2, 2 ], [ 2, 2 ] ];
    c.backward();
    console.log('grad b', b.grad.data);
    assert.isTrue(b.grad.data === -2.5);
  });
});
