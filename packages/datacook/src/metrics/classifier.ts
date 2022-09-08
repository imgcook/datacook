import { Tensor, Tensor1D, equal, sum, div, math, divNoNan, concat, mul, add, cast } from '@tensorflow/tfjs-core';
import { checkSameLength } from '../utils/validation';
import { getDiagElements } from '../linalg/utils';
import { LabelEncoder } from '../preprocess';

export type ClassificationReport = {
  precisions: Tensor;
  recalls: Tensor;
  f1s: Tensor;
  confusionMatrix: Tensor;
  categories: Tensor;
  accuracy: number;
  averagePrecision: number;
  averageRecall: number;
  averageF1: number;
};

export type ClassificationAverageTypes = 'macro' | 'weighted' | 'micro'

export const accuracyScore = (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): number => {
  const [ yTrueTensor, yPredTensor, nLabels ] = checkSameLength(yTrue, yPred);
  // TODO(sugarspectre): Accuaracy score computation
  const score = div(sum(equal(yPredTensor, yTrueTensor)), nLabels).dataSync()[0];
  return score;
};

export const getConfusionMatrix = async (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): Promise<{ confusionMatrix: Tensor, categories: Tensor }> => {
  const [ yTrueTensor, yPredTensor ] = checkSameLength(yTrue, yPred);
  const labelEncoder = new LabelEncoder();
  await labelEncoder.init(concat([ yTrueTensor, yPredTensor ]));
  const yTrueEncode = await labelEncoder.encode(yTrueTensor);
  const yPredEncode = await labelEncoder.encode(yPredTensor);
  const numClasses = labelEncoder.categories.shape[0];
  const confusionMatrix = cast(math.confusionMatrix(yTrueEncode as Tensor1D, yPredEncode as Tensor1D, numClasses), 'float32');
  return { confusionMatrix, categories: labelEncoder.categories };
};

const getPrecisionScoreByConfusionMatrix = (confusionMatrix: Tensor, average: ClassificationAverageTypes = 'micro'): number => {
  const confusionDiag = getDiagElements(confusionMatrix);
  const numClasses = confusionMatrix.shape[0];
  const precisions = divNoNan(confusionDiag, sum(confusionMatrix, 0));
  const weights = divNoNan(sum(confusionMatrix, 0), sum(confusionMatrix));
  const weightsSupport = divNoNan(sum(confusionMatrix, 1), sum(confusionMatrix));
  switch (average) {
  case 'micro':
    return sum(mul(precisions, weights)).dataSync()[0];
  case 'macro':
    return divNoNan(sum(precisions), numClasses).dataSync()[0];
  case 'weighted':
    return sum(mul(precisions, weightsSupport)).dataSync()[0];
  default:
    return sum(mul(precisions, weights)).dataSync()[0];
  }
};

const getRecallScoreByConfusionMatrix = (confusionMatrix: Tensor, average: ClassificationAverageTypes = 'micro'): number => {
  const confusionDiag = getDiagElements(confusionMatrix);
  const numClasses = confusionMatrix.shape[0];
  const recalls = divNoNan(confusionDiag, sum(confusionMatrix, 1));
  const weights = divNoNan(sum(confusionMatrix, 1), sum(confusionMatrix));
  switch (average) {
  case 'micro':
    return sum(mul(recalls, weights)).dataSync()[0];
  case 'macro':
    return divNoNan(sum(recalls), numClasses).dataSync()[0];
  case 'weighted':
    return sum(mul(recalls, weights)).dataSync()[0];
  default:
    return sum(mul(recalls, weights)).dataSync()[0];
  }
};

const getF1ScoreByConfusionMatrix = (confusionMatrix: Tensor, average: ClassificationAverageTypes = 'micro'): number => {
  const confusionDiag = getDiagElements(confusionMatrix);
  const numClasses = confusionMatrix.shape[0];
  const precisions = divNoNan(confusionDiag, sum(confusionMatrix, 0));
  const recalls = divNoNan(confusionDiag, sum(confusionMatrix, 1));
  const f1s = divNoNan(mul(mul(2, precisions), recalls), add(precisions, recalls));
  const weights = divNoNan(sum(confusionMatrix, 0), sum(confusionMatrix));
  const weightsSupport = divNoNan(sum(confusionMatrix, 1), sum(confusionMatrix));
  switch (average) {
  case 'micro': {
    const precision = sum(mul(precisions, weights)).dataSync()[0];
    const recall = sum(mul(recalls, weightsSupport)).dataSync()[0];
    return (divNoNan(mul(mul(2, precision), recall), add(precision, recall))).dataSync()[0];
  }
  case 'macro':
    return divNoNan(sum(f1s), numClasses).dataSync()[0];
  case 'weighted':
    return sum(mul(f1s, weightsSupport)).dataSync()[0];
  default:
    return sum(mul(f1s, weights)).dataSync()[0];
  }
};

/**
 * Compute the precision score for all classes.
 * Precision score is the ratio tp / (tp + fp), where tp is the number of true positives
 * and fp the number of false positives. The precision is intuitively the ability of
 * the classifier not to label as positive a sample that is negative.
 * The best value is 1 and the worst value is 0.
 * @param yTrue Ground truth (correct) target values.
 * @param yPred Estimated targets as returned by a classifier.
 * @returns Tensor of precision scores
 */
export const getPrecisionScores = async (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): Promise<{ precisions: Tensor, categories: Tensor }> => {
  const { confusionMatrix, categories } = await getConfusionMatrix(yTrue, yPred);
  const confusionDiag = getDiagElements(confusionMatrix);
  const precisions = divNoNan(confusionDiag, sum(confusionMatrix, 0));
  return { precisions: precisions, categories };
};

/**
 * Compute the recall score.
 * Recall score is the ratio tp / (tp + fn), where tp is the number of true positives
 * and fn the number of false negtive. The recall is intuitively the ability of the
 * classifier to find all the positive samples.
 * The best value is 1 and the worst value is 0.
 * @param yTrue Ground truth (correct) target values.
 * @param yPred Estimated targets as returned by a classifier.
 * @returns Tensor of recall scores
 */
export const getRecallScores = async (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): Promise<{ recalls: Tensor, categories: Tensor }> => {
  const { confusionMatrix, categories } = await getConfusionMatrix(yTrue, yPred);
  const confusionDiag = getDiagElements(confusionMatrix);
  return { recalls: divNoNan(confusionDiag, sum(confusionMatrix, 1)), categories: categories };
};

/**
 * Compute the f1 score.
 * The F1 score can be interpreted as a harmonic mean of the precision and recall,
 * where an F1 score reaches its best value at 1 and worst score at 0. The relative
 * contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
 * `2 * precision * recall / (precision + recall)`
  * @param yTrue Ground truth (correct) target values.
 * @param yPred Estimated targets as returned by a classifier.
 * @returns Tensor of f1 scores
 */
export const getF1Scores = async (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[]): Promise<{ f1s: Tensor, categories: Tensor }> => {
  const { confusionMatrix, categories } = await getConfusionMatrix(yTrue, yPred);
  const confusionDiag = getDiagElements(confusionMatrix);
  const precisions = divNoNan(confusionDiag, sum(confusionMatrix, 0));
  const recalls = divNoNan(confusionDiag, sum(confusionMatrix, 1));
  const f1s = divNoNan(mul(mul(2, precisions), recalls), add(precisions, recalls));
  return { f1s, categories };
};

/**
 * Compute the precision score.
 * Precision score is the ratio tp / (tp + fp), where tp is the number of true positives
 * and fp the number of false positives. The precision is intuitively the ability of
 * the classifier not to label as positive a sample that is negative.
 * The best value is 1 and the worst value is 0.
 * @param yTrue Ground truth (correct) target values.
 * @param yPred Estimated targets as returned by a classifier.
 * @param average \{'micro', 'macro', 'weighted'\}
 * This parameter is required for multiclass/multilabel targets. This determines the type of averaging
 * performed on the data, **default='micro'**:
 * - `'micro'`:
 *  Calculate metrics globally by counting the total true positives, false negatives and false positives.
 * - `'macro'`:
 *  Calculate metrics for each label, and find their unweighted mean. This does not take label
 *  imbalance into account.
 * - `'weighted'`:
 *  Calculate metrics for each label, and find their average weighted by support (the number of true
 * instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an
 * F-score that is not between precision and recall.
 * @returns Tensor of precision scores
 */
export const getPrecisionScore = async (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[], average: ClassificationAverageTypes = 'micro'): Promise<number> => {
  const { confusionMatrix } = await getConfusionMatrix(yTrue, yPred);
  return getPrecisionScoreByConfusionMatrix(confusionMatrix, average);
};


/**
 * Compute the recall score.
 * Recall score is the ratio tp / (tp + fn), where tp is the number of true positives
 * and fn the number of false negtive. The recall is intuitively the ability of the
 * classifier to find all the positive samples.
 * The best value is 1 and the worst value is 0.
 * @param yTrue Ground truth (correct) target values.
 * @param yPred Estimated targets as returned by a classifier.
 * @param average \{'micro', 'macro', 'weighted'\}
 * This parameter is required for multiclass/multilabel targets. This determines the type of averaging
 * performed on the data, **default='micro'**:
 * - `'micro'`:
 *  Calculate metrics globally by counting the total true positives, false negatives and false positives.
 * - `'macro'`:
 *  Calculate metrics for each label, and find their unweighted mean. This does not take label
 *  imbalance into account.
 * - `'weighted'`:
 *  Calculate metrics for each label, and find their average weighted by support (the number of true
 * instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an
 * F-score that is not between precision and recall.
 * @returns precision score
 */
export const getRecallScore = async (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[], average: ClassificationAverageTypes = 'micro'): Promise<number> => {
  const { confusionMatrix } = await getConfusionMatrix(yTrue, yPred);
  return getRecallScoreByConfusionMatrix(confusionMatrix, average);
};


/**
 * Compute the f1 score.
 * The F1 score can be interpreted as a harmonic mean of the precision and recall,
 * where an F1 score reaches its best value at 1 and worst score at 0. The relative
 * contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
 * `2 * precision * recall / (precision + recall)`
 * @param yTrue Ground truth (correct) target values.
 * @param yPred Estimated targets as returned by a classifier.
 * @param average \{'micro', 'macro', 'weighted'\}
 * This parameter is required for multiclass/multilabel targets. This determines the type of averaging
 * performed on the data, **default='micro'**:
 * - `'micro'`:
 *  Calculate metrics globally by counting the total true positives, false negatives and false positives.
 * - `'macro'`:
 *  Calculate metrics for each label, and find their unweighted mean. This does not take label
 *  imbalance into account.
 * - `'weighted'`:
 *  Calculate metrics for each label, and find their average weighted by support (the number of true
 * instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an
 * F-score that is not between precision and recall.
 * @returns precision score
 */
export const getF1Score = async (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[], average: ClassificationAverageTypes = 'micro'): Promise<number> => {
  const { confusionMatrix } = await getConfusionMatrix(yTrue, yPred);
  return getF1ScoreByConfusionMatrix(confusionMatrix, average);
};

/**
 * Generate classification report
 * @param yTrue true labels
 * @param yPred predicted labels
 * @returns classification report object, the struct of report will be like following
 */
export const getClassificationReport = async (yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[], average: ClassificationAverageTypes = 'weighted'): Promise<ClassificationReport> => {
  const { confusionMatrix, categories } = await getConfusionMatrix(yTrue, yPred);
  const confusionDiag = getDiagElements(confusionMatrix);
  const precisions = divNoNan(confusionDiag, sum(confusionMatrix, 0));
  const recalls = divNoNan(confusionDiag, sum(confusionMatrix, 1));
  const f1s = mul(divNoNan(mul(precisions, recalls), add(precisions, recalls)), 2);
  const accuracy = accuracyScore(yTrue, yPred);
  const averagePrecision = getPrecisionScoreByConfusionMatrix(confusionMatrix, average);
  const averageRecall = getRecallScoreByConfusionMatrix(confusionMatrix, average);
  const averageF1 = getF1ScoreByConfusionMatrix(confusionMatrix, average);
  return {
    precisions,
    recalls,
    f1s,
    confusionMatrix,
    categories,
    accuracy,
    averageF1,
    averagePrecision,
    averageRecall
  };
};
