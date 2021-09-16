import { Tensor, equal, sum, div, math, Tensor1D, divNoNan, concat, mul, add, cast, tidy } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';
import { getDiagElements } from '../linalg/utils';
import { LabelEncoder } from '../preprocess';

export type ClassificationReport = {
  precisions: Tensor;
  recalls: Tensor;
  f1s: Tensor;
  confusionMatrix: Tensor;
  classes: Tensor;
  accuracy: number;
  averagePrecision: number;
  averageRecall: number;
  averageF1: number;
};

export type ClassificationAverageTypes = 'macro' | 'weighted'

/**
 * Check that if yTrue and yPred are of same length
 * @param yTrue true labels
 * @param yPred predicted labels
 */
export const checkSameLength = (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): [ Tensor1D, Tensor1D, number ] => {
  const yTrueTensor = checkArray(yTrue, 'any', 1);
  const yPredTensor = checkArray(yPred, 'any', 1);
  const yTrueCount = yTrueTensor.shape[0];
  const yPredCount = yPredTensor.shape[0];
  if (yTrueCount != yPredCount) {
    throw new Error('Shape of yTrue should match shape of yPred');
  }
  return [ yTrueTensor as Tensor1D, yPredTensor as Tensor1D, yTrueCount ];
};

export const accuracyScore = (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): number => {
  const [ yTrueTensor, yPredTensor, nLabels ] = checkSameLength(yTrue, yPred);
  // TODO(sugarspectre): Accuaracy score computation
  const score = div(sum(equal(yPredTensor, yTrueTensor)), nLabels).dataSync()[0];
  return score;
};

/**
 * Generate classification report
 * @param yTrue true labels
 * @param yPred predicted labels
 * @returns classification report object, the struct of report will be like following
 */
export const classificationReport = async(yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[], average: ClassificationAverageTypes = 'weighted'): Promise<ClassificationReport> => {
  const [ yTrueTensor, yPredTensor ] = checkSameLength(yTrue, yPred);
  const labelEncoder = new LabelEncoder();
  await labelEncoder.init(concat([ yTrueTensor, yPredTensor ]));
  const yTrueEncode = await labelEncoder.encode(yTrueTensor);
  const yPredEncode = await labelEncoder.encode(yPredTensor);
  const numClasses = labelEncoder.categories.shape[0];
  const confusionMatrix = cast(math.confusionMatrix(yTrueEncode as Tensor1D, yPredEncode as Tensor1D, numClasses), 'float32');
  const confusionDiag = getDiagElements(confusionMatrix);
  const precisions = divNoNan(confusionDiag, sum(confusionMatrix, 0));
  const recalls = divNoNan(confusionDiag, sum(confusionMatrix, 1));
  const f1s = divNoNan(divNoNan(mul(precisions, recalls), add(precisions, recalls)), 2);
  const accuracy = accuracyScore(yTrue, yPred);
  const weights = divNoNan(sum(confusionMatrix, 0), sum(confusionMatrix));
  const averagePrecision = average == 'weighted' ? mul(precisions, weights).dataSync()[0] : divNoNan(sum(precisions), numClasses).dataSync()[0];
  const averageRecall = average == 'weighted' ? mul(recalls, weights).dataSync()[0] : divNoNan(sum(recalls), numClasses).dataSync()[0];
  const averageF1 = average == 'weighted' ? mul(f1s, weights).dataSync()[0] : divNoNan(sum(f1s), numClasses).dataSync()[0];
  return {
    precisions: precisions,
    recalls: recalls,
    f1s: f1s,
    confusionMatrix: confusionMatrix,
    classes: labelEncoder.categories,
    accuracy,
    averageF1,
    averagePrecision,
    averageRecall
  };
};
