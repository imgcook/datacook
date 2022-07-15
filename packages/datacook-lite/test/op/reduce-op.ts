// import { add2d, div2d, mul2d } from "../../src/core/op/binary-op";
import { matrix } from '../../src/core/classes';
import { assert } from 'chai';
import { matMul2d, sum2d } from "../../src/core/op";
import { scalar, vector } from "../../src/core/classes/creation";

describe('Reduce OP Test', () => {
  it('sum2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = sum2d(a);
    c.backward();
    console.log('c', c);
    console.log('c', c.data);
    console.log('grad c', c.grad.data);
    console.log('grad a', a.grad.data);
  });
});
