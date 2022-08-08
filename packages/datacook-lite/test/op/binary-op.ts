import { add2d, div2d, mul2d } from "../../src/core/op/binary-op";
import { matrix } from '../../src/core/classes';
import { assert } from 'chai';
import { matMul2d } from "../../src/core/op";
import { scalar, vector } from "../../src/core/classes/creation";
import { matrixEqual, scalarEqual, vectorEqual } from "../../src/utils/validation";

describe('OP Test', () => {
  it('add2d', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const c = add2d(a, 2);
    assert.isTrue(c.get(0, 0) === 3);
  });
  it('matmul2d', () => {
    const a = matrix([ [ 2, 3 ], [ 4, 5 ] ]);
    const b = matrix(([ [ -2, -3 ], [ -3, -5 ] ]));
    const c = matMul2d(a, b);
    c.backward();
    assert.isTrue(matrixEqual(c.grad, matrix([[1, 1], [1, 1]])));
    assert.isTrue(matrixEqual(a.grad, matrix([[-5, -8], [-5, -8]])));
    assert.isTrue(matrixEqual(b.grad, matrix([[6, 6],[8, 8]])));
  });
  it('add2d and matmul', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = scalar(2);
    const c = add2d(a, b);
    const d = matrix(([ [ -2, -3 ], [ -3, -5 ] ]));
    const e = matMul2d(c, d);
    e.backward();
    assert.isTrue(matrixEqual(d.grad, matrix([[8, 8], [10, 10]])));
    assert.isTrue(matrixEqual(c.grad, matrix([[-5, -8], [-5, -8]])));
    assert.isTrue(scalarEqual(b.grad, scalar(-26)));
    assert.isTrue(matrixEqual(a.grad, matrix([[-5, -8], [-5, -8]])));
  });

  it('mul', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = scalar(2);
    const c = mul2d(a, b);
    const gradA = [ [ 2, 2 ], [ 2, 2 ] ];
    c.backward();
    assert.isTrue(matrixEqual(a.grad, matrix(gradA)));
    assert.isTrue(scalarEqual(b.grad, scalar(10)));
  });
});

describe('OP div', () => {
  it('div2d matrix', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = matrix([ [ 1, 2 ], [ 1, 2 ] ]);
    const c = div2d(a, b);
    const gradA = [ [ -1, -0.5 ], [ -3, -1 ] ];
    c.backward();
    assert.isTrue(matrixEqual(a.grad, matrix(gradA)));
    
  });
  it('div2d vector', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = vector([ 1, 2 ]);
    const c = div2d(a, b);
    const gradB = [ -4, -1.5 ];
    c.backward();
    assert.isTrue(vectorEqual(b.grad, vector(gradB)));
  });
  it('div2d scalar', () => {
    const a = matrix([ [ 1, 2 ], [ 3, 4 ] ]);
    const b = scalar(2);
    const c = div2d(a, b);
    const gradA = [ [ 2, 2 ], [ 2, 2 ] ];
    c.backward();
    assert.isTrue(scalarEqual(b.grad, scalar(-2.5)));
  });
});
