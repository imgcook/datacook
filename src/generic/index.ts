import { Tensor } from '@tensorflow/tfjs-core';

function split(inputs: Tensor[], trainRatio = .75): Tensor[]{
  if (inputs.length === 0) {
    throw new Error('inputs should not have length of zero');
  }

  let size = inputs[0].shape[0];

  for (let i = 1; i < inputs.length; i++) {
    if (size !== inputs[i].shape[0]) {
      throw new Error('inputs should have the same length');
    }
  }

  const trainSize = Math.floor(size * trainRatio);
  const testSize = size - trainSize;

  const results = inputs.map((input) => input.split([ trainSize, testSize ])).reduce((prev, curr) => {
    prev.push(...curr);
    return prev;
  }, []);
  return results;
}

export * as npy from './npy';

export {
  split
};
