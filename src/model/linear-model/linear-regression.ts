import { Tensor, RecursiveArray, losses, squeeze, tensor, stack, tidy } from '@tensorflow/tfjs-core';
import { layers, sequential, Sequential, initializers, callbacks } from '@tensorflow/tfjs-layers';
import { BaseEstimater } from '../base';
import { checkArray } from '../../utils/validation';
import { Optimizer } from '@tensorflow/tfjs-core';
import { OptimizerProps, OptimizerType, getOptimizer } from '../../utils/optimizer-types';

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
  /**
   * Optimizer types
   */
  optimizerType?: OptimizerType,
  /**
   * Optimizer properties
   */
  optimizerProps?: OptimizerProps
}

export interface LinearRegerssionTrainParams {
  tol?: number,
  batchSize?: number,
  epochs?: number
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
  private optimizerType: OptimizerType;
  private optimizerProps: OptimizerProps;
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
   * `optimizerType`: optimizer types for training. All of the following optimizers types supported in tensorflow.js
   *  (https://js.tensorflow.org/api/latest/#Training-Optimizers) can be applied. Default to 'adam':
   *  - 'sgd': stochastic optimizer
   *  - 'momentum': momentum optimizer
   *  - 'adagrad': adagrad optimizer
   *  - 'adadelta': adadelta optimizer
   *  - 'adam': adam optimizer
   *  - 'adamax': adamax optimizer
   *  - 'rmsprop': rmsprop optimizer
   *
   * `optimizerProps`: parameters used to init corresponding optimizer, you can refer to documentations in tensorflow.js
   *  (https://js.tensorflow.org/api/latest/#Training-Optimizers) to find the supported initailization paratemters for a
   *  given type of optimizer. For example, `{ learningRate: 0.1, beta1: 0.1, beta2: 0.2, epsilon: 0.1 }` could be used
   *  to initialize adam optimizer.
   */
  constructor(params: LinearRegressionParams = {}) {
    super();
    this.fitIntercept = params.fitIntercept !== false;
    this.normalize = params.normalize;
    this.optimizerType = params.optimizerType ? params.optimizerType : 'adam';
    this.optimizerProps = params.optimizerProps ? params.optimizerProps : { learningRate: 0.1 };
    this.optimizer = getOptimizer(this.optimizerType, this.optimizerProps);
  }

  private initLinearModel(inputShape: number, useBias = true, weightsTensors: Tensor[] = []): void {
    const model = sequential();
    model.add(layers.dense({
      inputShape: [ inputShape ],
      units: 1,
      useBias,
      kernelInitializer: initializers.zeros(),
      biasInitializer: initializers.zeros(),
      kernelRegularizer: null
    }));
    model.compile({
      optimizer: this.optimizer,
      loss: losses.meanSquaredError,
      metrics: [ 'mse' ]
    });
    if (weightsTensors?.length) model.setWeights(weightsTensors);
    this.model = model;
  }

  /** Training logistic regression model by batch
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @returns classifier itself
   */
  public async trainOnBatch(xData: Tensor | RecursiveArray<number>, yData: Tensor | RecursiveArray<number>): Promise<LinearRegression> {
    const { x, y } = this.validateData(xData, yData);
    const nFeature = x.shape[1];
    if (this.model && nFeature !== this.featureSize) {
      throw new TypeError('feature size does not match previous training set');
    }
    if (!this.model) {
      this.initLinearModel(nFeature, this.fitIntercept);
      this.featureSize = nFeature;
    }
    await this.model.trainOnBatch(x, y);
    return this;
  }

  /** Fit linear regression model according to X, y. Here we use adam algorithm (a popular gradient-based optimization algorithm) for paramter estimation.
   * If `epochs` in parameter is not set, optimizer will iterate 10000 times at most (iteration will stop if converge).
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @param params batchSize: batch size: default to 32, epochs: epochs for training, default to Math.ceil(10000 / ( nData / batchSize ))
   * @returns classifier itself
   */
  public async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>,
    params: LinearRegerssionTrainParams = { batchSize: 32, tol: 0 }): Promise<LinearRegression> {

    const { x, y } = this.validateData(xData, yData);
    const nFeature = x.shape[1];
    const nData = x.shape[0];
    const batchSize = nData > params.batchSize ? params.batchSize : nData;
    const maxIterTimes = 10000;
    const defaultEpochs = Math.ceil(maxIterTimes / (nData / batchSize));
    const epochs = params.epochs > 0 ? params.epochs : defaultEpochs;
    const tol = params.tol ? params.tol : 0;
    this.initLinearModel(nFeature, this.fitIntercept);
    this.featureSize = nFeature;
    await this.model.fit(x, y, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: callbacks.earlyStopping({ monitor: 'loss', minDelta: tol })
    });
    return this;
  }

  public async predict(xData: Tensor | RecursiveArray<number>): Promise<Tensor> {
    return tidy(() => {
      const x = checkArray(xData, 'float32');
      const predY = this.model.predict(x);
      if (predY instanceof Tensor) {
        return squeeze(predY);
      } else {
        return squeeze(stack(predY));
      }
    });
  }

  public initModelFromWeights(inputShape: number, useBias: boolean, weights: (Float32Array | Int32Array | Uint8Array)[]): void {
    const weightsTensors = [];
    for (const w of weights) {
      weightsTensors.push(tensor(w));
    }
    this.initLinearModel(inputShape, useBias, weightsTensors);
  }

  /**
   * Get linear regression coefficients
   * @returns Linear regression coefficients with structure
   * {'coefficient': Tensor, 'intercept': Tensor}
   */
  public getCoef(): { coefficients: Tensor, intercept: Tensor } {
    return {
      coefficients: squeeze(this.model.getWeights()[0]),
      intercept: this.fitIntercept ? this.model.getWeights()[1] : tensor(0)
    };
  }

  public getModelWeightsArray(): Promise<RecursiveArray<number>> {
    return Promise.all(this.model.getWeights().map((w: Tensor) => w.array()));
  }

  /**
   * Load model paramters from json string object
   * @param modelJson model json saved as string object
   * @returns model itself
   */
  public async fromJson(modelJson: string): Promise<LinearRegression> {
    const modelParams = JSON.parse(modelJson);
    if (modelParams.name !== 'LinearRegression') {
      throw new TypeError(`${modelParams.name} is not Linear Regression`);
    }
    const { fitIntercept, optimizerType, optimizerProps,
      modelWeights, featureSize } = modelParams;
    this.fitIntercept = !!fitIntercept;
    this.featureSize = featureSize;
    if (optimizerType && optimizerProps) {
      this.optimizer = getOptimizer(optimizerType, optimizerProps);
      this.optimizerType = optimizerType;
      this.optimizerProps = optimizerProps;
    }
    if (modelWeights?.length) {
      this.initModelFromWeights(featureSize, fitIntercept, modelWeights);
    }
    return this;
  }

  /**
   * Dump model parameters to json string.
   * @returns Stringfied model parameters
   */
  public async toJson(): Promise<string> {
    const modelParams = {
      name: 'LinearRegression',
      fitIntercept: this.fitIntercept,
      optimizerType: this.optimizerType,
      optimizerProps: this.optimizerProps,
      modelWeights: await this.getModelWeightsArray(),
      featureSize: this.featureSize
    };
    return JSON.stringify(modelParams);
  }
}
