import { Tensor, RecursiveArray, losses, train, squeeze, tensor } from '@tensorflow/tfjs-core';
import { layers, sequential, Sequential, regularizers, callbacks } from '@tensorflow/tfjs-layers';
import { checkArray } from '../../utils/validation';
import { BaseClassifier } from '../base';

// export enum LogisticPenalty {
//   L1 = 'l1',
//   L2 = 'l2',
//   None = 'none'
// }

export type LogisticPenalty = 'l1' | 'l2' | 'none';

/**
 * Parameters for linear regression
 */
export interface LogisticRegressionParams {
  /**
   * {'l1', 'l2', 'none'}, default = 'l2'
   * Specify the norm used in the penalization.
   */
  penalty?: LogisticPenalty,
  /**
   * Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
   */
  fitIntercept?: boolean,
  /**
   * This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
   */
  normalize?: boolean,
  /**
   * Regularization strength; must be a positive float. Larger values specify stronger regularization.
   * Default is 1
   */
  c?: number
}

export interface LogisticRegressionTrainParams {
  /**
   * Tolerence for stopping criteria, default is 1e-4
   */
  tol?: number,
  learningRate?: number,
  batchSize?: number,
  maxIterTimes?: number,
}

const defaultLearningTrainParams = {
  tol: 0,
  learningRate: 0.5,
  batchSize: 32,
  maxIterTimes: 20000
};

/**
 * Logistic regression classifier
 * */
export class LogisticRegression extends BaseClassifier {
  private fitIntercept: boolean;
  private penalty: LogisticPenalty;
  private c: number;
  private model: Sequential;
  private featureSize: number;
  private outputSize: number;
  /**
   * Construction function of linear regression model
   * @param params LinearRegressionParams
   */
  constructor(params: LogisticRegressionParams = {
    penalty: 'l2',
    fitIntercept: true,
    c: 1
  }) {
    super();
    this.fitIntercept = params.fitIntercept !== false;
    this.penalty = params.penalty;
    this.c = params.c;
  }

  private initModel(inputShape: number, outputShape: number, useBias = true): Sequential {
    const model = sequential();
    const c = this.c;
    const penalty = this.penalty;
    model.add(layers.dense({
      inputShape: [ inputShape ],
      units: outputShape,
      useBias,
      activation: 'sigmoid',
      kernelRegularizer: penalty == 'l2' ? regularizers.l2({ l2: c }) : penalty == 'l1' ? regularizers.l1({ l1: c }) : null }));
    return model;
  }

  /** Train batch data
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @param params batch size, default to 32
   * @returns classifier itself
   */
  // public async trainBatch(xData: Tensor | RecursiveArray<number>, yData: Tensor | RecursiveArray<number>, params: LogisticRegressionParams = { batchSize: 32 }): Promise<LogisticRegression> {
  //   const { x, y } = this.validateData(xData, yData);
  //   const nFeature = x.shape[1];
  //   if (!this.model) {
  //     this.initClasses(y);
  //     this.model = this.initModel(nFeature, this.classes.shape[0], this.fitIntercept);
  //     this.featureSize = nFeature;
  //   } else {
  //     if (nFeature != this.featureSize) {
  //       throw new Error('feature size does not match previous training set');
  //     }
  //   }
  //   await this.model.fit(x, y, {
  //     batchSize: params.batchSize,
  //     epochs: 1,
  //     shuffle: true
  //   });
  //   return this;
  // }

  /** Training linear regression model according to X, y. Here we use adam algorithm (a popular gradient-based optimization algorithm) for paramter estimation.
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @param params batchSize: batch size: default to 32, maxIterTimes: max iteration times, default to 20000
   * @returns classifier itself
   */
  public async train(xData: Tensor | RecursiveArray<number>, yData: Tensor | RecursiveArray<number>, params: LogisticRegressionTrainParams = defaultLearningTrainParams): Promise<LogisticRegression> {
    const { x, y } = this.validateData(xData, yData);
    this.initClasses(y);
    const nFeature = x.shape[1];
    const nData = x.shape[0];
    const batchSize = nData > params.batchSize ? params.batchSize : nData;
    const epochs = Math.ceil(params.maxIterTimes / nData);
    const outputShape = this.isBinaryClassification() ? 1 : this.classes.shape[0];
    this.model = this.initModel(nFeature, outputShape, this.fitIntercept);
    this.model.compile({
      optimizer: train.adam(params.learningRate),
      loss: losses.sigmoidCrossEntropy
    });
    this.featureSize = nFeature;
    const yOneHot = this.getLabelOneHot(y);
    await this.model.fit(x, yOneHot, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: callbacks.earlyStopping({ monitor: 'loss', minDelta: 0 })
    });
    return this;
  }

  public predict(xData: Tensor | RecursiveArray<number>) : Tensor | Tensor[] {
    const x = checkArray(xData, 'float32');
    const scores = this.model.predict(x);
    //const predY = this.getPredClass(scores);
    if (scores instanceof Tensor) {
      return this.getPredClass(scores);
    }
    return scores;
  }

  public getCoef(): { 'coefficients': Tensor, 'intercept': Tensor} {
    return {
      'coefficients': squeeze(this.model.getWeights()[0]),
      'intercept': this.fitIntercept ? this.model.getWeights()[1] : tensor(0)
    };
  }
}
