import { Tensor, equal, sum, div } from '@tensorflow/tfjs-core';

export const accuracyScore = ( yTrue: Tensor, yPred: Tensor ): number => {
  const yTrueCount = yTrue.shape[0];
  const yPredCount = yPred.shape[0];
  if (yTrueCount != yPredCount) {
    throw new Error('Shape of yTrue should match shape of yPred');
  }
  // TODO(sugarspectre): Accuaracy score computation
  const score = div(sum(equal(yPred, yTrue)), yTrueCount).dataSync()[0];
  return score;
};
