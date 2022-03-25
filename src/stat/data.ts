import { Tensor, RecursiveArray, sub, mean, divNoNan, sum, pow, sqrt, transpose, topk, stack, slice, neg, tidy } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';
import { isInteger } from '../math/utils';

/**
 * Get centered data.
 * centered_X = X - mean(X)
 * @param xData input data
 * @param axis centering axis
 * @returns centered data
 */
export const getCenteredData = (xData: Tensor | RecursiveArray<number>, axis = -1): Tensor => {
  return tidy(() => {
    const xTensor = checkArray(xData, 'float32');
    const dim = xTensor.shape.length;
    if (dim === 1) {
      return sub(xTensor, mean(xTensor));
    }
    if (!isInteger(axis) || axis >= dim || axis < -1) {
      throw new TypeError(`Invalid axis: ${axis}`);
    }
    if (axis === -1) {
      return sub(xTensor, mean(xTensor));
    }
    if (axis === 0) {
      return sub(xTensor, mean(xTensor, axis));
    }
    // transpose permutation
    const perm: number[] = [];
    const inversePerm: number[] = [];
    for (let i = 0; i < dim; i++) {
      const permIdx = i === 0 ? axis : i <= axis ? i - 1 : i;
      perm[i] = permIdx;
      inversePerm[permIdx] = i;
    }
    return transpose(sub(transpose(xTensor, perm), mean(xTensor, axis)), inversePerm);
  });
};

export const getVarianceFromCentered = (xCentered: Tensor, axis: number): Tensor => {
  return tidy(() => {
    const dim = xCentered.shape.length;
    if (dim === 1) {
      const nSamples = xCentered.shape[0];
      return divNoNan(sum(pow(xCentered, 2)), nSamples - 1);
    }
    if (!isInteger(axis) || axis >= dim || axis < -1) {
      throw new TypeError(`Invalid axis: ${axis}`);
    }
    if (axis === -1) {
      let nSamples = 1;
      xCentered.shape.forEach((d: number) => {
        nSamples = nSamples * d;
      });
      return divNoNan(sum(pow(xCentered, 2)), sub(nSamples, 1));
    }
    if (axis === 0) {
      return divNoNan(sum(pow(xCentered, 2), axis), xCentered.shape[0] - 1);
    }
    const nSamples = xCentered.shape[axis];
    return divNoNan(sum(pow(xCentered, 2), axis), nSamples - 1);
  });
};

/**
 * Calculate variance of input data.\
 * variance = (X - mean(X))'(X - mean(X)) / (n - 1)
 * @param xData input data
 * @param axis axis for calculating variance, **default = -1**, which means calculation will be
 * applied across all axes.
 * @returns tensor of data variance
 */
export const getVariance = (xData: Tensor | RecursiveArray<number>, axis = -1): Tensor => {
  return tidy(() => {
    const xTensor = checkArray(xData, 'float32');
    const xCentered = getCenteredData(xTensor, axis);
    return getVarianceFromCentered(xCentered, axis);
  });
};

export const getMean = (xData: Tensor | RecursiveArray<number>, axis = -1): Tensor => {
  const xTensor = checkArray(xData, 'float32');
  return mean(xTensor, axis);
};

/**
 * Standardize input data.\
 * standard_x = (X - mean(X)) / sqrt(var(X))
 * @param xData Input data
 * @param axis axis for standardize, **default = -1**, which means standardization will be
 * applied across all axes.
 * @returns data after standardization
 */
export const standardize = (xData: Tensor | RecursiveArray<number>, axis = -1): Tensor => {
  return tidy(() => {
    const xTensor = checkArray(xData, 'float32');
    const xCentered = getCenteredData(xTensor, axis);
    const xStd = sqrt(getVarianceFromCentered(xCentered, axis));
    const dim = xTensor.shape.length;
    if (dim === 1 || axis === -1 || axis === 0) {
      return divNoNan(xCentered, xStd);
    }
    // transpose permutation
    const perm: number[] = [];
    const inversePerm: number[] = [];
    for (let i = 0; i < dim; i++) {
      const permIdx = i === 0 ? axis : i <= axis ? i - 1 : i;
      perm[i] = permIdx;
      inversePerm[permIdx] = i;
    }
    return transpose(divNoNan(transpose(xTensor, perm), xStd), inversePerm);
  });
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
