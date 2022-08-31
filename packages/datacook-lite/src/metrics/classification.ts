import { sum1d } from "../backend-cpu/op";
import { equal1d } from "../backend-cpu/op/binary-op";
import { vector, Vector } from "../core/classes";

export const accuracyScore = (yTrue: Array<any> | Vector, yPred: Array<any> | Vector): number => {
  const yTrueCount = yTrue.length;
  const yPredCount = yPred.length;
  const yTrueVector = yTrue instanceof Array ? vector(yTrue) : yTrue;
  const yPredVector = yPred instanceof Array ? vector(yPred) : yPred;
  if (yTrueCount != yPredCount) {
    throw new Error('Shape of yTrue should match shape of yPred');
  }
  return sum1d(equal1d(yTrueVector, yPredVector)).values() / yTrueCount;
};
