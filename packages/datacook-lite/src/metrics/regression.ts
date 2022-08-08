import { sum1d, sub1d, mean1d, square1d } from "../core/op";
import { vector } from "../core/classes";
/**
 * Computation of R-square value.
 * R^2 = 1 - sum((yTrue - yPred)^2) / sum((yTrue - mean(yTrue))^2)
 * @param yTrue true values
 * @param yPred prediced values
 * @returns r-square value
 */
export const getRSquare = (yTrue: number[], yPred: number[]): number => {
  //   const { yTrueTensor, yPredTensor } = checkPairInput(yTrue, yPred);
  const yTrueVector = vector(yTrue);
  const yPredVector = vector(yPred);
  const numerator = sum1d(square1d(sub1d(yTrueVector, yPredVector)));
  const denom = sum1d(square1d(sub1d(yTrueVector, mean1d(yTrueVector))));
  return 1.0 - numerator / denom;
};
