import { incbeta, beta } from '../../../src/math/beta';
import { assert } from 'chai';
import { numEqual } from '../../../src/math/utils';

describe('Beta', () => {
  it('calculate beta function', () => {
    const a = beta(5, 6);
    const b = beta(10, 3);
    // compare with result calculated in scipy
    assert.isTrue(numEqual(a, 0.0007936507936507937, 1e-3));
    assert.isTrue(numEqual(b, 0.0015151515151515152, 1e-3));
  });
  it('calculate incomplete beta function', () => {
    const a = incbeta(2.2, 3.1, 0.4);
    const b = incbeta(10, 10, 0.3);
    const c = incbeta(10, 10, 0.5);
    const d = incbeta(10, 10, 0.7);
    const e = incbeta(10, 10, 1);
    // compare with result calculated in scipy
    assert.isTrue(numEqual(a, 0.49339638807619446, 1e-3));
    assert.isTrue(numEqual(b, 0.03255335688130093, 1e-3));
    assert.isTrue(numEqual(c, 0.5000000000000002, 1e-3));
    assert.isTrue(numEqual(d, 0.967446643118699, 1e-3));
    assert.isTrue(numEqual(e, 1, 1e-3));
  });
});
