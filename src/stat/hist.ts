import { Tensor1D } from "@tensorflow/tfjs-core";
import { checkJSArray } from "../utils/validation";

/**
 * Get histogram data for given input
 * @param xData inpur data of shape [nSamples, ]
 * @param params parameters
 * Option in params
 * ----
 * `bins`: number of cells, **default=50**
 * `leftLim`: left limit to calculate, **default=min value of input data**
 * `rightLim`: right limit to calculate, **default=max value of input data**
 * @returns
 * { steps; number[], counts: number[] }
 * `steps`: array of giving the breakpoints between histogram cells,
 * `counts`: array of number of data points in histogram cells
 */
export const getHistData = (xData: Tensor1D | number[], params?: {
    bins?: number,
    leftLim?: number,
    rightLim?: number
  }): {
    steps: number[],
    counts: number[],
  } => {
  const xArray = checkJSArray(xData, 'float32', 1) as number[];
  const nData = xArray.length;
  const minVal = Math.min(...xArray);
  const maxVal = Math.max(...xArray);
  const { bins = 50, leftLim = minVal, rightLim = maxVal } = params;
  if (rightLim < leftLim) {
    throw new TypeError('Right limit should be greater than left limit');
  }
  const step = (rightLim - leftLim) / bins;
  const steps = Array.from(new Array(bins).keys()).map((i) => leftLim + step * i);
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < nData; i++) {
    if (xArray[i] < leftLim || xArray[i] > rightLim) {
      continue;
    }
    if (xArray[i] === rightLim) {
      counts[ bins - 1 ] += 1;
    } else {
      counts[ Math.floor((xArray[i] - leftLim) / step) ] += 1;
    }
  }
  return {
    steps,
    counts
  };
};
