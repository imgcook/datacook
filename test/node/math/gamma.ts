import { gamma, lnGamma } from '../../../src/math/gamma';
import { numEqual } from '../../../src/math/utils';
import { assert } from 'chai';

describe('Gamma', () => {
  it('calculate gamma function', () => {
    const a = gamma(0);
    const b = gamma(1);
    const c = gamma(2);
    const d = gamma(3);
    const e = gamma(5);  
    const f = gamma(5.5);
    const aTrue = Infinity;
    const bTrue = 1;
    const cTrue = 1;
    const dTrue = 2.0;
    const eTrue = 24;
    const fTrue = 52.3427;
    assert.isTrue(a === aTrue);
    assert.isTrue(Math.abs(b - bTrue) < 1e-3);
    assert.isTrue(Math.abs(c - cTrue) < 1e-3);
    assert.isTrue(Math.abs(d - dTrue) < 1e-3);
    assert.isTrue(Math.abs(e - eTrue) < 1e-3);
    assert.isTrue(Math.abs(f - fTrue) < 1e-3);
  });
  it('culculate ln gamma function', () => {
    const a = lnGamma(0);
    const b = lnGamma(0.03);
    const c = lnGamma(0.2);
    const d = lnGamma(0.5);
    const e = lnGamma(5);  
    const f = lnGamma(5.5);
    const g = lnGamma(10);
    assert.isTrue(a === Infinity);
    assert.isTrue(numEqual(b, 3.48997104, 1e-4));
    assert.isTrue(numEqual(c, 1.52406382, 1e-4));
    assert.isTrue(numEqual(d, 0.57236494, 1e-4));
    assert.isTrue(numEqual(e, 3.17805383, 1e-4));
    assert.isTrue(numEqual(f, 3.95781397, 1e-4));
    assert.isTrue(numEqual(g, 12.80182748, 1e-4));
  });
});
