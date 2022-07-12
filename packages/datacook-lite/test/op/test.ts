import { add2d } from "../../src/core/op/binary-op";
import { matrix } from '../../src/core/classes';
import { assert } from 'chai';
import { matMul2d } from "../../src/core/op";

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
});
