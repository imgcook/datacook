import { getHistData } from '../../../src/stat';
import { assert } from 'chai';

describe('Get Histogram data', () => {
  it('simple random test', () => {
    const x = Array.from(new Array(5000).keys()).map(() => Math.abs((Math.random())));
    const hist = getHistData(x, { bins: 50 });
    assert.isTrue(hist.steps.length === 50);
    assert.isTrue(hist.counts.reduce((a, b) => a + b) === 5000);
  });
});
