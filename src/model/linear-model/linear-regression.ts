import { Tensor, RecursiveArray, losses, train, squeeze, tensor } from '@tensorflow/tfjs-core';
import { layers, sequential, Sequential } from '@tensorflow/tfjs-layers';
import { BaseEstimater } from '../base';
import { checkArray } from '../../utils/validation';
import { Optimizer } from '@tensorflow/tfjs-core';

/**
 * Parameters for linear regression
 */
export interface LinearRegressionParams {
  /**
   * Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
   */
  fitIntercept?: boolean, 
  /**
   * This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
   */
  normalize?: boolean,
  optimizer?: Optimizer
}

export interface LinearRegerssionTrainParams {
  tol?: number,
  batchSize?: number,
  epochs?: number,
}

/**
 * Linear regression model
 * LinearRegression fits a linear model with coefficients w = (w1, …, wp)
 * to minimize the residual sum of squares between the observed targets in
 * the dataset, and the targets predicted by the linear approximation.
 */
export class LinearRegression extends BaseEstimater {
  private fitIntercept: boolean;
  private normalize: boolean;
  private model: Sequential;
  private featureSize: number;
  private optimizer: Optimizer;
  /**
   * Construction function of linear regression model
   * @param params LinearRegressionParams
   *
   * Options in params
   * ------------
   *
   * `fitIntercept`: Whether to calculate the intercept for this model. If set to False,
   * no intercept will be used in calculations
   *
   * `normalize`: This parameter is ignored when fit_intercept is set to False. If True,
   * the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
   *
   * `optimizer`: Optimizer for training. All of the optimizers in tensorflow.js (https://js.tensorflow.org/api/latest/#Training-Optimizers)
   *  can be applied. Default to adam optimzer with learning rate 0.1.
   */
  constructor(params: LinearRegressionParams = {}) {
    super();
    this.fitIntercept = params.fitIntercept !== false;
    this.normalize = params.normalize;
    this.optimizer = params.optimizer && params.optimizer instanceof Optimizer ? params.optimizer : train.adam(0.1);
  }

  private initLinearModel(inputShape: number, useBias = true): Sequential {
    const model = sequential();
    model.add(layers.dense({ inputShape: [ inputShape ], units: 1, useBias }));
    model.compile({
      optimizer: this.optimizer,
      loss: losses.meanSquaredError,
      metrics: [ 'mse' ]
    });
    return model;
  }

  /** Training logistic regression model by batch
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @returns classifier itself
  */
  public async trainOnBatch(xData: Tensor | RecursiveArray<number>, yData: Tensor | RecursiveArray<number>): Promise<LinearRegression> {
    const { x, y } = this.validateData(xData, yData);
    const nFeature = x.shape[1];
    if (!this.model) {
      this.model = this.initLinearModel(nFeature, this.fitIntercept);
      this.featureSize = nFeature;
    } else {
      if (nFeature != this.featureSize) {
        throw new Error('feature size does not match previous training set');
      }
    }
    await this.model.trainOnBatch(x, y);
    return this;
  }

  /** Training linear regression model according to X, y. Here we use adam algorithm (a popular gradient-based optimization algorithm) for paramter estimation.
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @param params batchSize: batch size: default to 32, maxIterTimes: max iteration times, default to 20000
   * @returns classifier itself
   */
  public async train(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>,
    params: LinearRegerssionTrainParams = { batchSize: 32, epochs: -1 }): Promise<LinearRegression> {

    const { x, y } = this.validateData(xData, yData);
    const nFeature = x.shape[1];
    const nData = x.shape[0];
    const batchSize = nData > params.batchSize ? params.batchSize : nData;
    const epochs = params.epochs > 0 ? params.epochs : null;
    this.model = this.initLinearModel(nFeature, this.fitIntercept);
    this.featureSize = nFeature;
    await this.model.fit(x, y, {
      batchSize,
      epochs,
      shuffle: true
    });
    return this;
  }

  public predict(xData: Tensor | RecursiveArray<number>): Tensor | Tensor[] {
    const x = checkArray(xData, 'float32');
    const predY = this.model.predict(x);
    return predY;
  }

  public getCoef(): { 'coefficients': Tensor, 'intercept': Tensor} {
    return {
      'coefficients': squeeze(this.model.getWeights()[0]),
      'intercept': this.fitIntercept ? this.model.getWeights()[1] : tensor(0)
    };
  }
}
