import { checkArray } from '../utils/validation';
import { Tensor, RecursiveArray, min, max, tidy, squeeze, dispose, Tensor1D } from '@tensorflow/tfjs-core';

export interface HistData {
  from: number;
  to: number;
  count: number;
}

/**
 * Generate histogram Data especially for data visualization.
 * @param xData input data
 * @param bins split bin counts, default to 50
 * @param limitLeft limit value for left side
 * @param limitRight limit value for right side
 */
export const genHistData = async (xData: Tensor | RecursiveArray<number>, bins = 50,
  limitLeft?: number, limitRight?: number): Promise<HistData[]> => {
  const xTensor = tidy(() => squeeze(checkArray(xData, 'float32')) as Tensor1D);
  const minVal = limitLeft ? limitLeft : tidy(() => min(xTensor).dataSync())[0];
  const maxVal = limitRight ? limitRight : tidy(() => max(xTensor).dataSync())[0];
  if (minVal > maxVal) {
    throw new TypeError('Maximum value should be greater than minimum value');
  }
  const xArray = await xTensor.array();
  const histData = new Array(bins).fill(0);
  const binStep = (maxVal - minVal) / bins;
  for (let i = 0; i < xArray.length; i++) {
    const histPos = Math.floor(xArray[i] / binStep);
    histData[histPos] += 1;
  }
  dispose(xTensor);
  return histData.map((d: number, i: number) => {
    return {
      from: minVal + i * binStep,
      to: minVal + (i + 1) * binStep - 1,
      count: d
    };
  });
};
