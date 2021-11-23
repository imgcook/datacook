import { losses } from "@tensorflow/tfjs-core";
import { LossOrMetricFn } from "@tensorflow/tfjs-layers/dist/types";

export type InitailizerType = 'constant'|'glorotNormal'|'glorotUniform'|
  'heNormal'|'heUniform'|'identity'| 'leCunNormal'|'leCunUniform'|'ones'|
  'orthogonal'|'randomNormal'| 'randomUniform'|'truncatedNormal'|'varianceScaling'|
  'zeros';

export type LossType = 'mse' | 'mae' | 'hinge' | 'log' | 'sigmoid' | 'softmax';

export const getLossFunction = (lossType: LossType): LossOrMetricFn => {
  switch (lossType) {
  case 'mse':
    return losses.meanSquaredError;
  case 'mae':
    return losses.absoluteDifference;
  case 'hinge':
    return losses.hingeLoss;
  case 'log':
    return losses.logLoss;
  case 'sigmoid':
    return losses.sigmoidCrossEntropy;
  case 'softmax':
    return losses.softmaxCrossEntropy;
  default:
    return losses.meanSquaredError;
  }
};
