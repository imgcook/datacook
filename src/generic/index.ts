import { Tensor } from '@tensorflow/tfjs-core';
import 'seedrandom';


function shuffle(inputs: Array<any>, seed?: string): void {
  if (!seed) seed = Math.random().toString();
  // @ts-ignore
  const rng = new Math.seedrandom(seed);
  for (let i = inputs.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [ inputs[i], inputs[j] ] = [ inputs[j], inputs[i] ];
  }
}

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
  split,
  shuffle
};
