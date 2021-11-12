import { cdf } from '../../../src/stat/t';
import { assert } from 'chai';
import { numEqual } from '../../../src/math/utils';

describe('T Distribution', () => {
  it('calculate cumulative density function', () => {
    const p1 = cdf(0.1, 2);
    const p2 = cdf(0, 2);
    const p3 = cdf(0.1, 5);
    const p4 = cdf(0.3, 5);
    const p5 = cdf(0.5, 5);
    const p6 = cdf(0.9, 5);
    const p7 = cdf(5, 5);
    const p8 = cdf(10000, 1000);
    const p9 = cdf(0.1, 200);
    // compare with result calculated in scipy
    assert.isTrue(numEqual(p1, 0.53526728079293, 1e-3));
    assert.isTrue(numEqual(p2, 0.5, 1e-3));
    assert.isTrue(numEqual(p3, 0.5378849294226699, 1e-3));
    assert.isTrue(numEqual(p4, 0.6118754788683627, 1e-3));
    assert.isTrue(numEqual(p5, 0.6808505641795355, 1e-3));
    assert.isTrue(numEqual(p6, 0.7953143998276881, 1e-3));
    assert.isTrue(numEqual(p7, 0.9979476420099733, 1e-3));
    assert.isTrue(numEqual(p8, 1, 1e-3));
    assert.isTrue(numEqual(p9, 0.5397777537478394, 1e-3));
  });
});
