import { Tensor, RecursiveArray, sub, mean, transpose, matMul, divNoNan } from '@tensorflow/tfjs-core';
import { normalize } from './data';
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
  const xNormalized = normalize(xData);
  return getCovarianceMatrix(xNormalized);
};
