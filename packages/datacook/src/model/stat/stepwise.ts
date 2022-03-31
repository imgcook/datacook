import { booleanMaskAsync, RecursiveArray, Tensor } from '@tensorflow/tfjs-core';
import { LinearRegressionAnalysis } from './linear-regression-analysis';
import { checkArray } from '../../utils/validation';

export interface StepwiseLinearRegressionParams {
  featureNames?: string[],
}

export const stepwiseLinearRegression = async (xData: Tensor | RecursiveArray<number>,
  yData: Tensor | RecursiveArray<number>, featureNames?: string[]): Promise<LinearRegressionAnalysis> => {
  const xTensor = checkArray(xData, 'float32', 2);
  const yTensor = checkArray(yData, 'float32', 1);
  const featureSize = xTensor.shape[1];
  let curXTensor = xTensor;
  let minAIC = Number.MAX_SAFE_INTEGER;
  let minLm: LinearRegressionAnalysis;
  let nDelFeatures = 0;
  const curFeatureNames = [ ...featureNames ];
  let fitIntercept = true;
  for (let i = 0; i < featureSize; i++) {
    const lm = new LinearRegressionAnalysis({ fitIntercept: !!fitIntercept });
    await lm.fit(curXTensor, yTensor, curFeatureNames);
    const summary = lm.summary();
    const coefs = summary.coefficients;
    let maxP = 0;
    if (summary.aic < minAIC) {
      minAIC = summary.aic;
      minLm = lm;
    }
    let delIndex = -1;
    for (let i = 0; i < coefs.length; i++) {
      if (maxP < coefs[i].pValue && coefs[i].pValue > 0.05) {
        maxP = coefs[i].pValue;
        delIndex = i;
      }
    }
    if (delIndex === -1) {
      return minLm;
    } else {
      if (delIndex !== 0) {
        const mask = (Array.from(new Array(featureSize - nDelFeatures))).map((v: any, i: number) => delIndex - 1 !== i);
        curXTensor = await booleanMaskAsync(curXTensor, mask, 1);
        nDelFeatures += 1;
        if (curFeatureNames) {
          curFeatureNames.splice(delIndex - 1, 1);
        }
      } else {
        fitIntercept = false;
      }
    }
  }
};
