import { Tensor } from '@tensorflow/tfjs-core';
import 'seedrandom';

function seed(seed: string): void {
  (Math as any).seedrandom(seed);
}

function shuffle(inputs: Array<any>): void {
  for (let i = inputs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
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

function range(start: number, end: number, step?: number): Array<number> {
  if (!step) step = 1;

  if ((step > 0 && start > end) || (step < 0 && start < end)) return [];

  const length = Math.floor(Math.abs(end - start) / Math.abs(step));
  const arr = new Array(length);

  let x = start;
  let idx = 0;

  /**
   * continue adding elements
   * if step > 0 and x < end or step < 0  and x > end
   */
  while ((step > 0 && x < end) || (step < 0 && x > end)) {
    arr[idx] = x;
    idx += 1;
    x += step;
  }

  return arr;
}

export * as npy from './npy';

export {
  split,
  shuffle,
  seed,
  range
};
