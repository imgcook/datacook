import { Tensor, RecursiveArray, sub, mean, divNoNan, sum, pow, sqrt, square } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';

/**
 * Calculate variance of input data.\
 * In standard mode, variance = (X - mean(X))'(X - mean(X)) / (n - 1)\
 * In common mode, variance = (X - mean(X))'(X - mean(X)) / n
 * @param xData input data
 * @param axis axis for computing, 1: by row, 0: by column, -1: by all. Default is 0
 * @param standrad if computed in standard mode. Default is true.
 * @returns tensor of data variance
 * TODO: support multi-dimentional computation for variance
 */
export const getVariance = (xData: Tensor | RecursiveArray<number>, axis = 0, standard = true): Tensor => {
  const xTensor = checkArray(xData, 'float32');
  if (xTensor.shape.length > 2) {
    throw new TypeError('Tree dimensional data is currently not supported');
  }
  if (xTensor.shape.length === 1) {
    const nSamples = xTensor.shape[0];
    const xCentered = sub(xTensor, mean(xTensor));
    const nDiv = standard ? nSamples - 1 : nSamples;
    return divNoNan(sum(pow(xCentered, 2)), nDiv);
  } else {
    const nDiv = axis === 0 ? xTensor.shape[0] : axis === 1 ? xTensor.shape[1] : xTensor.shape[0] * xTensor.shape[1];
    const xCentered = sub(xTensor, mean(xTensor, axis));
    if (standard) {
      return divNoNan(sum(square(xCentered), axis), nDiv -1);
    }
    else {
      return divNoNan(sum(square(xCentered), axis), nDiv);
    }
  }
};

/**
 * Calculate variance of input data.\
 * variance = (X - mean(X))'(X - mean(X)) / n
 * @param xData input data
 * @returns tensor of data variance
 */
 export const getVarianceStandard = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const axisV = 0;
  const xTensor = checkArray(xData, 'float32');
  const nSamples = xTensor.shape[0];
  const xCentered = sub(xTensor, mean(xTensor, axisV));
  return divNoNan(sum(pow(xCentered, 2), axisV), nSamples);
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
