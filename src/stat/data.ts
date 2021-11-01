import { Tensor, sub, mean, divNoNan, sum, pow, sqrt, RecursiveArray } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';

/**
 * Calculate variance of input data.\
 * variance = (X - mean(X))'(X - mean(X)) / (n - 1)
 * @param xData input data
 * @returns tensor of data variance
 */
export const variance = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const axisV = 0;
  const xTensor = checkArray(xData, 'float32');
  const nSamples = xTensor.shape[0];
  const xCentered = sub(xTensor, mean(xTensor, axisV));
  return divNoNan(sum(pow(xCentered, 2), axisV), nSamples - 1);
};

/**
 * Normalize input data.\
 * normalized_x = (X - mean(X)) / sqrt(var(X))
 * @param xData Input data
 * @returns tensor of normalized data
 */
export const normalize = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const axisV = 0;
  const xTensor = checkArray(xData, 'float32');
  const xCentered = sub(xTensor, mean(xTensor, axisV));
  const xStd = sqrt(variance(xTensor));
  return divNoNan(xCentered, xStd);
};
