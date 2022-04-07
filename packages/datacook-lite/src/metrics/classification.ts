export const accuracyScore = (yTrue: Array<any>, yPred: Array<any>): number => {
  const yTrueCount = yTrue.length;
  const yPredCount = yPred.length;
  if (yTrueCount != yPredCount) {
    throw new Error('Shape of yTrue should match shape of yPred');
  }
  return yTrue.reduce((s, y, i: number): number => i === 0 ? Number(y === yPred[i]) : (Number(y === yPred[i]) + s)) * 1.0 / yTrueCount;
};
