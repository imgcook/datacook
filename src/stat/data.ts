import { Tensor, RecursiveArray, sub, mean, divNoNan, sum, pow, sqrt, transpose, topk, stack, slice, neg } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';

/**
 * Calculate variance of input data.\
 * variance = (X - mean(X))'(X - mean(X)) / (n - 1)
 * @param xData input data
 * @returns tensor of data variance
 */
export const getVariance = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const axisV = 0;
  const xTensor = checkArray(xData, 'float32');
  const nSamples = xTensor.shape[0];
  const xCentered = sub(xTensor, mean(xTensor, axisV));
  return divNoNan(sum(pow(xCentered, 2), axisV), nSamples - 1);
};

export const getMean = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const axisV = 0;
  const xTensor = checkArray(xData, 'float32');
  return mean(xTensor, axisV);
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
  const xStd = sqrt(getVariance(xTensor));
  return divNoNan(xCentered, xStd);
};

/**
 * Calculate quantile data of input.
 * @param xData input data
 */
export const quantile = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const xTensor = checkArray(xData, 'float32');
  const quantiles = [ 0, 0.25, 0.5, 0.75, 1 ];
  if (xTensor.shape.length > 2) {
    throw new TypeError('Three dimensional data is currently not supported');
  }
  const xTensorTranspose = xTensor.shape.length === 2 ? transpose(xTensor) : xTensor;
  const xTensorTransposeNeg = neg(xTensorTranspose);
  const nData = xTensorTranspose.shape[0];
  const isTwoDimension = xTensor.shape.length === 2;
  const quantileNumbers: Array<Tensor> = [];
  for (let i = 0; i < quantiles.length; i++) {
    const isRightQuantile = i > 0.5;
    const quantileNumber = isRightQuantile ? (quantiles[i] * nData) : ((1 - quantiles[i]) * nData);
    const quantileIndex = Math.ceil(quantileNumber) + 1;
    const { values } = isRightQuantile ? (topk(xTensorTransposeNeg, quantileIndex, true)) : (topk(xTensorTranspose, quantileIndex, true));
    const isExact = quantileNumber === Math.floor(quantileNumber);
    const sliceNumber = isExact ? 1 : 2;
    if (isTwoDimension){
      if (isRightQuantile)
        quantileNumbers.push(mean(slice(values, [ 0, 0 ], [ -1, sliceNumber ])));
    } else {
      quantileNumbers.push(mean(slice(values, 0, sliceNumber)));
    }
  }
  return stack(quantileNumbers);
};
