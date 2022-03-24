import { Tensor, RecursiveArray, sub, mean, transpose, matMul, divNoNan, Tensor1D, mul, sqrt, sum } from '@tensorflow/tfjs-core';
import { getVariance, normalize } from './data';
import { checkArray } from '../utils/validation';

/**
 * Get covariance matrix for given input matrix with shape (n, m)
 * If X and Y are two random variables, with means (expected values) μX and μY
 * and standard deviations σX and σY, respectively, then their covariance is as follow:\
 * covariance = E[(X - μX)(Y - μY)]
 * @param xData input data of shape (n, m)
 * @returns covariance matrix with shape (m, m)
 */
export const getCovarianceMatrix = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const axisV = 0;
  const xTensor = checkArray(xData, 'float32', 2);
  const nSamples = xTensor.shape[0];
  const xCentered = sub(xTensor, mean(xTensor, axisV));
  return divNoNan(matMul(transpose(xCentered), xCentered), nSamples - 1);
};

/**
 * Get correlation matrix for given input matrix with shape (n, m)
 * If X and Y are two random variables, with means (expected values) μX and μY
 * and standard deviations σX and σY, respectively, then their correlation is as follow:\
 * correlation = E[(X - μX)(Y - μY)] / (σXσY)
 * @param xData input data of shape (n, m)
 * @returns correlation matrix with shape with shape (m, m)
 */
export const getCorrelationMatrix = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const xTensor = checkArray(xData, 'float32', 2);
  const xNormalized = normalize(xTensor, 0);
  return getCovarianceMatrix(xNormalized);
};

/**
 * Get covariance for given input x and y with shape (n,)
 * If X and Y are two random variables, with means (expected values) μX and μY
 * and standard deviations σX and σY, respectively, then their covariance is as follow:\
 * covariance = E[(X - μX)(Y - μY)]
 * @param x input array x of shape (n,)
 * @param y input array y of shape (n,)
 * @returns covariance of x and y
 */
export const getCovariance = (x: Tensor1D | number[], y: Tensor1D | number[]): number => {
  const xTensor = checkArray(x, 'float32', 1);
  const yTensor = checkArray(y, 'float32', 1);
  if (xTensor.shape[0] !== yTensor.shape[0]) {
    throw new TypeError('Length of input array x and y should be the same');
  }
  if (xTensor.shape[0] === 1) {
    throw new TypeError('Length of input array should be greater than 1');
  }
  const nData = xTensor.shape[0];
  const xMean = mean(x);
  const yMean = mean(y);
  return divNoNan(sum(mul(sub(xTensor, xMean), sub(yTensor, yMean))), nData - 1).dataSync()[0];
};

/**
 * Get correlation for given input x and y with shape (n,)
 * If X and Y are two random variables, with means (expected values) μX and μY
 * and standard deviations σX and σY, respectively, then their correlation is as follow:\
 * correlation = E[(X - μX)(Y - μY)] / (σXσY)
 * @param x input array x of shape (n,)
 * @param y input array y of shape (n,)
 * @returns correltaion of x and y
 */
export const getCorrelation = (x: Tensor1D | number[], y: Tensor1D | number[]): number => {
  const xTensor = checkArray(x, 'float32', 1) as Tensor1D;
  const yTensor = checkArray(y, 'float32', 1) as Tensor1D;
  const covariance = getCovariance(xTensor, yTensor);
  const xVariance = getVariance(x);
  const yVariance = getVariance(y);
  return divNoNan(covariance, sqrt(mul(xVariance, yVariance))).dataSync()[0];
};
