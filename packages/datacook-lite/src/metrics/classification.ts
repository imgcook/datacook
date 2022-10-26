import { Vector } from "../core/classes";

export const accuracyScore = (yTrue: Array<any> | Vector, yPred: Array<any> | Vector): number => {
  const yTrueCount = yTrue.length;
  const yPredCount = yPred.length;
  const yTrueArr = yTrue instanceof Vector ? yTrue.values() : yTrue;
  const yPredArr = yPred instanceof Vector ? yPred.values() : yPred;
  if (yTrueCount != yPredCount) {
    throw new Error('Shape of yTrue should match shape of yPred');
  }
  return yTrueArr.filter((d, i) => d == yPredArr[i]).length / yTrueCount;
};
