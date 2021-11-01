import { Tensor, sub, mean, transpose, matMul, divNoNan, RecursiveArray } from '@tensorflow/tfjs-core';
import { normalize } from './data';
import { checkArray } from '../utils/validation';

export const covarianceMatrix = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const axisV = 0;
  const xTensor = checkArray(xData, 'float32', 2);
  const nSamples = xTensor.shape[0];
  const xCentered = sub(xTensor, mean(xTensor, axisV));
  return divNoNan(matMul(transpose(xCentered), xCentered), nSamples - 1);
};

export const correlationMatrix = (xData: Tensor | RecursiveArray<number>): Tensor => {
  const xNormalized = normalize(xData);
  return covarianceMatrix(xNormalized);
};
