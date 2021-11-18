import { cdf } from '../../../src/stat/normal';
import { assert } from 'chai';
import { numEqual } from '../../../src/math/utils';

describe('Normal Distribution', () => {
  it('calculate cumulative density function', () => {
    const p1 = cdf(0.1);
    const p2 = cdf(0.2);
    const p3 = cdf(0.5);
    const p4 = cdf(1000);
    const p5 = cdf(-0.1);
    const p6 = cdf(-0.2);
    const p7 = cdf(-0.5);
    const p8 = cdf(-1000);
    // compare with result calculated in scipy
    assert.isTrue(numEqual(p1, 0.539827837277029, 1e-3));
    assert.isTrue(numEqual(p2, 0.579259709439103, 1e-3));
    assert.isTrue(numEqual(p3, 0.6914624612740131, 1e-3));
    assert.isTrue(numEqual(p4, 1, 1e-3));
    assert.isTrue(numEqual(p5, 0.460172162722971, 1e-3));
    assert.isTrue(numEqual(p6, 0.42074029056089696, 1e-3));
    assert.isTrue(numEqual(p7, 0.3085375387259869, 1e-3));
    assert.isTrue(numEqual(p8, 0, 1e-3));
  });
});
