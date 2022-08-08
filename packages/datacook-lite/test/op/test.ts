import { add2d } from "../../src/core/op/binary-op";
import { matrix } from '../../src/core/classes';
import { assert } from 'chai';

describe('OP Test', () => {
  it('add2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = add2d(a, 2);
    assert(c.get(0, 0) === 3);
  });
});
