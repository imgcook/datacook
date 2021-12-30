import { Tensor1D, tidy, sum, square, sub, mean, divNoNan, Tensor } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';
import { shapeEqual } from '../linalg';

export const checkPairInput = (yTrue: Tensor | number[], yPred: Tensor | number[]): { yTrueTensor: Tensor, yPredTensor: Tensor } => {
  return tidy(() => {
    const yTrueTensor = checkArray(yTrue, 'any', 1) as Tensor1D;
    const yPredTensor = checkArray(yPred, 'any', 1) as Tensor1D;
    if (!shapeEqual(yTrueTensor, yPredTensor)) {
      throw new Error('Shape of yTrue should match shape of yPred');
    }
    return { yTrueTensor, yPredTensor };
  });
};

/**
 * Computation of R-square value.
 * R^2 = 1 - sum((yTrue - yPred)^2) / sum((yTrue - mean(yTrue))^2)
 * @param yTrue true labels
 * @param yPred prediced labels
 * @returns r-square value
 */
export const getRSquare = (yTrue: Tensor1D | number[], yPred: Tensor1D | number[]): number => {
  return tidy(() => {
    const { yTrueTensor, yPredTensor } = checkPairInput(yTrue, yPred);
    const numerator = sum(square(sub(yTrueTensor, yPredTensor)));
    const denom = sum(square(sub(yTrueTensor, mean(yTrueTensor))));
    return 1.0 - divNoNan(numerator, denom).dataSync()[0];
  });
};

/**
 * Computation of mean squared error.
 * MSE = 1 /n * sum((yTrue - yPred)^2)
 * @param yTrue true labels
 * @param yPred predicted labels
 * @returns mean squared error
 */
export const getMeanSquaredError = (yTrue: Tensor1D | number[], yPred: Tensor1D | number[]): number => {
  return tidy(() => {
    const { yTrueTensor, yPredTensor } = checkPairInput(yTrue, yPred);
    const nData = yTrueTensor.shape[0];
    return 1.0 / nData * sum(square(sub(yTrueTensor, yPredTensor))).dataSync()[0];
  });
};

/**
 * Copmutation of residual variance
 * var(residual) = 1 / (n - df(model)) * sum((yTrue - yPred)^2)
 * @param yTrue true labels
 * @param yPred predicted labels
 * @returns residual variance
 */
export const getResidualVariance = (yTrue: Tensor | number[], yPred: Tensor | number[], paramSize: number): number => {
  return tidy(() => {
    const { yTrueTensor, yPredTensor } = checkPairInput(yTrue, yPred);
    const nData = yTrueTensor.shape[0];
    return 1.0 / (nData - paramSize) * sum(square(sub(yTrueTensor, yPredTensor))).dataSync()[0];
  });
};

/**
 * Compute adjuested r-square.
 * RSquare_adjusted = 1 - [((1 - RSquare) * (n - 1)) / (n - k - 1)]
 * where n is the number of data samples.
 * k is the number of independent regressors,
 * i.e. the number of variables in your model, excluding the constant.
 * @param yTrue true output
 * @param yPred predicted ouput
 * @param k number of independent regressors
 * @returns adjusted r-square
 */
export const getAdjustedRSquare = (yTrue: Tensor | number[], yPred: Tensor | number[], k: number): number => {
  return tidy(() => {
    const { yTrueTensor, yPredTensor } = checkPairInput(yTrue, yPred);
    const rSquare = getRSquare(yTrueTensor, yPredTensor);
    const nData = yTrueTensor.shape[0];
    return 1 - ((1 - rSquare) * (nData - 1)) / (nData - k - 1);
  });
};
