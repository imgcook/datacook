import { tensor1d, memory } from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import { tidyAsync } from '../../../src/utils/memory';

describe('utils', function () {
  it('creates a tidy and check if Promise is supported.', async () => {
    await tidyAsync(() => {
      return new Promise<number>((resolve) => setTimeout(resolve.bind(null, 1), 500));
    });
  });

  it('adds an async ops to a created tidy and check its memory management.', async () => {
    const tensorBase = memory().numTensors;
    const bytesBase = memory().numBytes;
    console.log('the before memory info is', tensorBase, bytesBase);
    await tidyAsync(async () => {
      const a = tensor1d([1, 2, 3, 4]);
      const b = tensor1d([2, 3, 4, 5]);
      await a.data();
      await b.data();
      assert.equal(memory().numTensors, tensorBase + 2);
      assert.equal(memory().numBytes, bytesBase + 32);
    });
    assert.equal(memory().numTensors, tensorBase);
    assert.equal(memory().numBytes, bytesBase);
  });

  it('adds an async ops with exceptions and check its memory management', async () => {
    const tensorBase = memory().numTensors;
    try {
      await tidyAsync(async () => {
        tensor1d([0, 4, 2, 2]);
        tensor1d([0, 4, 2, 3]);
        assert.equal(memory().numTensors, tensorBase + 2);
        throw new TypeError('this is an exception.');
        assert.fail('should not be reachable');
      });
    } catch (err) {
      assert.equal(err.message, 'this is an exception.');
    }
    assert.equal(memory().numTensors, tensorBase);
  });

  it('adds multiple calls to tidy, and got an error when first not ended', async () => {
    tidyAsync(async () => {
      return new Promise<number>((resolve) => setTimeout(resolve.bind(null, 1), 500)); 
    });

    let caughtError = false;
    try {
      await tidyAsync(async () => tensor1d([0]));
    } catch (err) {
      assert.equal(err.message,
        'tidy must have only 1 running instance, please create a new tidy after others end.');
      caughtError = true;
    }
    assert.equal(caughtError, true);
  });
});

