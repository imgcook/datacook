// import { Tensor, RecursiveArray, sub, mean, divNoNan, sum, pow, sqrt, transpose, topk, stack, slice, neg, tidy } from '@tensorflow/tfjs-core';
import { div2d, square2d, sqrt1d } from "../backend-cpu/op";
import { matrix, Matrix, Vector } from "../core/classes";
import { sub2d, mean2d, sum2d, div1d } from "../core/op";

/**
 * Get centered data.
 * centered_X = X - mean(X)
 * @param x input data
 * @param axis centering axis
 * @returns centered data
 */
export const getCenteredData = (x: Matrix, axis = -1): Matrix => {
  return sub2d(x, mean2d(x, axis), axis);
};

export const getVarianceFromCentered = (xCentered: Matrix, axis: number): number | Vector => {
  const [ nX, mX ] = xCentered.shape;
  const squaredX = square2d(xCentered);
  if (axis === -1) {
    return (sum2d(squaredX) as number) / (nX * mX - 1);
  }
  if (axis === 0) {
    return div1d((sum2d(squaredX, 0) as Vector), nX - 1);
  }
  if (axis === 1) {
    return div1d((sum2d(squaredX, 1) as Vector), mX - 1);
  }
};

/**
 * Calculate variance of input data.\
 * variance = (X - mean(X))'(X - mean(X)) / (n - 1)
 * @param xData input data
 * @param axis axis for calculating variance, **default = -1**, which means calculation will be
 * applied across all axes.
 * @returns tensor of data variance
 */
export const getVariance = (xData: number[][], axis = -1): Vector | number => {
  const xMatrix = matrix(xData);
  const xCentered = getCenteredData(xMatrix, axis);
  return getVarianceFromCentered(xCentered, axis);
};

export const getMean = (xData: number[][], axis = -1): Vector | number => {
  // const xTensor = checkArray(xData, 'float32');
  const xMatrix = matrix(xData);
  return mean2d(xMatrix, axis);
};

/**
 * Standardize input data.\
 * standard_x = (X - mean(X)) / sqrt(var(X))
 * @param xData Input data
 * @param axis axis for standardize, **default = -1**, which means standardization will be
 * applied across all axes.
 * @returns data after standardization
 */
export const standardize = (xData: Matrix, axis = -1): Matrix => {
  // const xTensor = checkArray(xData, 'float32');
  const xCentered = getCenteredData(xData, axis);
  if (axis === 1 || axis === 0) {
    const xStd = sqrt1d(getVarianceFromCentered(xCentered, axis) as Vector);
    return div2d(xCentered, xStd, axis);
  } else {
    const xStd = Math.sqrt(getVarianceFromCentered(xCentered, -1) as number);
    return div2d(xCentered, xStd);
  }
};
