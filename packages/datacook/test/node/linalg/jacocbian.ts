import { tensor, Variable, variable, scalar, Tensor } from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { getJacobian } from '../../../src/linalg/jacobian';
import { assert } from 'chai';
import 'mocha';

describe('jacobian Solver', () => {
  it('solve jacobian matrix', async () => {
    const expr = (tf: any, x: Tensor, a: Variable, b: Variable) => tf.add(tf.mul(a, tf.sqrt(x)), b);
    const a = variable(scalar(4));
    const b = variable(scalar(5));
    const x = tensor([ 1, 2, 3, 4 ]);
    const { jacobian } = getJacobian(expr, x, a, b);
    const jacTrue = [ 1, 2, 3, 4 ].map((d) => [ Math.sqrt(d), 1 ]);
    assert.isTrue(tensorEqual(tensor(jacTrue), jacobian, 1e-3));
  });
});
