import { Tensor1D } from "@tensorflow/tfjs-core";
import { checkJSArray } from "../utils/validation";

/**
 * Get histogram data for given input
 * @param xData inpur data of shape [nSamples, ]
 * @param params parameters
 * Option in params
 * ----
 * `bins`: number of cells, **default=50**
 * `leftLimit`: left limit to calculate, **default=min value of input data**
 * `rightLimit`: right limit to calculate, **default=max value of input data**
 * @returns
 * { steps; number[], counts: number[] }
 * `steps`: array of giving the breakpoints between histogram cells,
 * `counts`: array of number of data points in histogram cells
 */
export const getHistData = (xData: Tensor1D | number[], params?: {
    bins?: number,
    leftLimit?: number,
    rightLimit?: number
  }): {
    steps: number[],
    counts: number[],
  } => {
  const xArray = checkJSArray(xData, 'float32', 1) as number[];
  const nData = xArray.length;
  const minVal = Math.min(...xArray);
  const maxVal = Math.max(...xArray);
  const { bins = 50, leftLimit = minVal, rightLimit = maxVal } = params;
  if (rightLimit < leftLimit) {
    throw new TypeError('Right limit should be greater than left limit');
  }
  const step = (rightLimit - leftLimit) / bins;
  const steps = Array.from(new Array(bins).keys()).map((i) => leftLimit + step * i);
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < nData; i++) {
    if (xArray[i] < leftLimit || xArray[i] > rightLimit) {
      continue;
    }
    if (xArray[i] === rightLimit) {
      counts[ bins - 1 ] += 1;
    } else {
      counts[ Math.floor((xArray[i] - leftLimit) / step) ] += 1;
    }
  }
  return {
    steps,
    counts
  };
};
