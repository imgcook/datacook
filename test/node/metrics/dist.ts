
import { getKLDivergence, getJSDivergence } from '../../../src/metrics/dist';
import { numEqual } from '../../../src/math/utils';
import { assert } from 'chai';

describe('Dist', () => {
  it('get kl divergence', () => {
    const p = [ 0.1, 0.2, 0.3, 0.4, 0.5 ];
    const q = [ 0.2, 0.3, 0.5, 0.6, 0.7 ];
    const kl = getKLDivergence(p, q);
    assert.isTrue(numEqual(kl, 0.004725622586074958, 1e-4));
  });
  it('get js divergence', () => {
    const p = [ 0.1, 0.2, 0.3, 0.4, 0.5 ];
    const q = [ 0.2, 0.3, 0.5, 0.6, 0.7 ];
    const js = getJSDivergence(p, q);
    assert.isTrue(numEqual(js, 0.004834588863737572, 1e-4));
  });
});
