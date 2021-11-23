import { Tensor, RecursiveArray, losses, squeeze, tensor, stack, sigmoid, softmax } from '@tensorflow/tfjs-core';
import { layers, sequential, Sequential, regularizers, callbacks, initializers } from '@tensorflow/tfjs-layers';
import { Optimizer } from '@tensorflow/tfjs-core';

import { checkArray } from '../../utils/validation';
import { BaseClassifier } from '../base';
import { OptimizerProps, OptimizerType, getOptimizer } from '../../utils/optimizer-types';

export type LogisticPenalty = 'l1' | 'l2' | 'none';
/**
 * Parameters for linear regression:
 */
export interface LogisticRegressionParams {
  /**
   * {'l1', 'l2', 'none'}, default = 'none'
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
  c?: number,
  /**
   * Optimizer types
   */
  optimizerType?: OptimizerType,
  /**
   * Optimizer properties
   */
  optimizerProps?: OptimizerProps
}

export interface LogisticRegressionTrainParams {
  tol?: number,
  batchSize?: number,
  epochs?: number,
}
/**
 * Logistic regression classifier
 */
export class LogisticRegression extends BaseClassifier {
  private fitIntercept: boolean;
  private penalty: LogisticPenalty;
  private c: number;
  private model: Sequential;
  private featureSize: number;
  private outputSize: number;
  private optimizerType: OptimizerType;
  private optimizerProps: OptimizerProps;
  private optimizer: Optimizer;
  /**
   * Construction function of linear regression model
   * @param params LinearRegressionParams
   *
   * Options in `params`
   * ---------
   *
   * `penalty`: {'l1', 'l2', 'none'}, default to 'none', Specify the norm used in the penalization.
   *
   * `fitIntercept`: Whether to calculate the intercept for this model. If set to False,
   *    no intercept will be used in calculations.
   *
   * `c`: Regularization strength; must be a positive float. Larger values specify stronger regularization.
   * Default to 1.
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
  constructor(params: LogisticRegressionParams = {
    penalty: 'none',
    fitIntercept: true,
    c: 1,
    optimizerType: 'adam',
    optimizerProps: { learningRate: 0.1 }
  }) {
    super();
    this.fitIntercept = params.fitIntercept !== false;
    this.penalty = params.penalty ? params.penalty : 'none';
    this.c = params.c ? params.c : 1;
    this.optimizerType = params.optimizerType ? params.optimizerType : 'adam';
    this.optimizerProps = params.optimizerProps ? params.optimizerProps : { learningRate: 0.1 };
    this.optimizer = getOptimizer(this.optimizerType, this.optimizerProps);
  }

  private initModel(inputShape: number, outputShape: number, useBias = true, modelWeights: Tensor[] = []): void {
    const model = sequential();
    const c = this.c;
    const penalty = this.penalty;
    model.add(layers.dense({
      inputShape: [ inputShape ],
      units: outputShape,
      useBias,
      kernelInitializer: initializers.zeros(),
      biasInitializer: initializers.zeros(),
      kernelRegularizer: penalty === 'l2' ? regularizers.l2({ l2: c }) : penalty === 'l1' ? regularizers.l1({ l1: c }) : null
    }));
    model.compile({
      optimizer: this.optimizer,
      loss: this.isBinaryClassification() ? losses.sigmoidCrossEntropy : losses.softmaxCrossEntropy
    });
    if (modelWeights.length > 0) {
      model.setWeights(modelWeights);
    }
    this.model = model;
  }

  /** Training logistic regression model by batch
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @returns classifier itself
   */
  public async trainOnBatch(xData: Tensor | RecursiveArray<number>, yData: Tensor | RecursiveArray<number>): Promise<LogisticRegression> {
    const { x, y } = this.validateData(xData, yData);
    const nFeature = x.shape[1];
    if (this.model && nFeature != this.featureSize) {
      throw new Error('feature size does not match previous training set');
    }
    if (!this.model) {
      if (!this.classes() || !this.classes().shape[0]) await this.initClasses(y, 'binary-only');
      const outputShape = this.isBinaryClassification() ? 1 : this.classes().shape[0];
      this.initModel(nFeature, outputShape, this.fitIntercept);
      this.featureSize = nFeature;
      this.outputSize = outputShape;
    }
    await this.model.trainOnBatch(x, await this.getLabelOneHot(y));
    return this;
  }

  /** Fit linear regression model according to X, y.
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @param params training parameters batchSize: batch size: default to 32, maxIterTimes: max iteration times, default to 20000
   * @returns classifier itself
   *
   * Options in `params`
   * --------------
   *
   * `batchSize`: training batch size, default to 32
   *
   * `epochs`: training epochs, default to -1, which means training will not stop until converge
   *
   * `tol`: stop tolerence, default to 0
   */
  public async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>, params: LogisticRegressionTrainParams = {
      tol: 0,
      batchSize: 32,
      epochs: -1
    }): Promise<LogisticRegression> {

    const { x, y } = this.validateData(xData, yData);
    await this.initClasses(y, 'binary-only');
    const nFeature = x.shape[1];
    const nData = x.shape[0];
    const batchSize = nData > params.batchSize ? params.batchSize : nData;
    const epochs = params.epochs > 0 ? params.epochs : null;
    const outputShape = this.isBinaryClassification() ? 1 : this.classes().shape[0];
    this.initModel(nFeature, outputShape, this.fitIntercept);
    this.featureSize = nFeature;
    this.outputSize = outputShape;
    const yOneHot = await this.getLabelOneHot(y);

    await this.model.fit(x, yOneHot, {
      batchSize,
      epochs,
      shuffle: true,
<<<<<<< HEAD
      callbacks: callbacks.earlyStopping({ monitor: 'loss', minDelta: 0 })
=======
      callbacks: callbacks.earlyStopping({ monitor: 'loss', minDelta: params.tol })
>>>>>>> a204a8a299a83931037e6addd72fe7816251b3e4
    });
    return this;
  }

  /**
   * Make predictions using logistic regression model.
   * @param xData Input features
   * @returns Predicted classes
   */
  public async predict(xData: Tensor | RecursiveArray<number>): Promise<Tensor> {
    const x = checkArray(xData, 'float32');
    const scores = this.model.predict(x);
    if (scores instanceof Tensor) {
      return this.getPredClass(scores);
    } else {
      return this.getPredClass(stack(scores));
    }
  }

  /**
   * Predict probabilities using logistic regression model.
   * @param xData Input features
   * @returns Predicted probabilities
   */
  public async predictProba(xData: Tensor | RecursiveArray<number>): Promise<Tensor> {
    const x = checkArray(xData, 'float32');
    const scores = this.model.predict(x);
    if (scores instanceof Array){
      return this.isBinaryClassification() ? stack(scores) : softmax(stack(scores));
    } else {
      return this.isBinaryClassification() ? sigmoid(scores) : softmax(scores);
    }
  }

  public getCoef(): { coefficients: Tensor, intercept: Tensor } {
    return {
      coefficients: squeeze(this.model.getWeights()[0]),
      intercept: this.fitIntercept ? this.model.getWeights()[1] : tensor(0)
    };
  }

  public getModelWeightsArray(): Promise<RecursiveArray<number>> {
    return Promise.all(this.model.getWeights().map((w: Tensor) => w.array()));
  }

  public initModelFromWeights(inputShape: number, outputShape: number, useBias: boolean, weights: (Float32Array | Int32Array | Uint8Array)[]): void {
    const weightsTensors = weights.map((w) => tensor(w));
    this.initModel(inputShape, outputShape, useBias, weightsTensors);
  }

  /**
   * Load model paramters from json string object
   * @param modelJson model json saved as string object
   * @returns model itself
   */
  public async fromJson(modelJson: string): Promise<LogisticRegression> {
    const modelParams = JSON.parse(modelJson);
    if (modelParams.name !== 'LogisticRegression') {
      throw new TypeError(`${modelParams.name} is not Logistic Regression`);
    }
    const { classes, fitIntercept, penalty, c, optimizerType, optimizerProps,
      modelWeights, featureSize, outputSize } = modelParams;
    if (classes) {
      this.initClasses(classes, 'binary-only');
    }
    this.fitIntercept = (fitIntercept as boolean);
    if (penalty) {
      this.penalty = penalty;
    }
    if (c) {
      this.c = c;
    }
    if (optimizerType as OptimizerType && optimizerProps as OptimizerProps) {
      this.optimizerType = optimizerType;
      this.optimizerProps = optimizerProps;
      this.optimizer = getOptimizer(optimizerType, optimizerProps);
    }
    if (modelWeights?.length) {
      this.initModelFromWeights(featureSize, outputSize, fitIntercept, modelWeights);
    }
    return this;
  }
  /**
   * Dump model parameters to json string.
   * @returns Stringfied model parameters
   */
  public async toJson(): Promise<string> {
    const modelParams = {
      name: 'LogisticRegression',
      classes: await this.classes()?.array(),
      fitIntercept: this.fitIntercept,
      penalty: this.penalty,
      c: this.c,
      optimizerType: this.optimizerType,
      optimizerProps: this.optimizerProps,
      modelWeights: await this.getModelWeightsArray(),
      featureSize: this.featureSize,
      outputSize: this.outputSize
    };
    return JSON.stringify(modelParams);
  }
}
