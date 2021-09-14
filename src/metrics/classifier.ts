import { Tensor, equal, sum, div } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';

export const accuracyScore = ( yTrue: Tensor | string[] | number[], yPred: Tensor | string[] | number[] ): number => {
  const yTrueTensor = checkArray(yTrue, 'any', 1);
  const yPredTensor = checkArray(yPred, 'any', 1);
  const yTrueCount = yTrueTensor.shape[0];
  const yPredCount = yPredTensor.shape[0];
  if (yTrueCount != yPredCount) {
    throw new Error('Shape of yTrue should match shape of yPred');
  }
  // TODO(sugarspectre): Accuaracy score computation
  const score = div(sum(equal(yPred, yTrue)), yTrueCount).dataSync()[0];
  return score;
};

/**
 * Check that yTrue and yPred belong to classification task
 * @param yTrue 
 * @param yPred 
 */
export const checkTarget = (yTrue, yPred) {
  
}

export const confusionMatrix = (yTrue: Tensor | string[] | number[], yPred, sampleWeight) {

}
