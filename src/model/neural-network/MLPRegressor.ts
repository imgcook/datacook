import { Tensor, RecursiveArray, Optimizer, squeeze, tensor, stack } from '@tensorflow/tfjs-core';
import { callbacks, layers, sequential, Sequential, CustomCallback } from '@tensorflow/tfjs-layers';
import { BaseRegressor } from '../base';
import { checkArray } from '../../utils/validation';
import { OptimizerProps, OptimizerType, getOptimizer } from '../../utils/optimizer-types';
import { InitailizerType, LossType, getLossFunction } from '../../utils/sgd-types';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

// export type MLPRegressorWeightInitializer = ''

export interface MLPRegressorParams {
  hiddenLayerSizes?: number[],
  /**
   * Optimizer types
   */
  optimizerType?: OptimizerType,
  /**
   * Optimizer properties
   */
  optimizerProps?: OptimizerProps,
  /**
   * hidden layer activation type
   */
  activations?: ActivationIdentifier | ActivationIdentifier[],
  /**
   * loss function of training, 'mse' and 'mae' can be choosen
   */
  loss?: LossType,
  /**
   * weight initializer
   */
  initializer?: InitailizerType
}

export interface MLPRegressorTrainParams {
  tol?: number,
  batchSize?: number,
  epochs?: number,
  // verbose?: ModelLoggingVerbosity,
  earlyStopping?: boolean,
  verbose?: boolean
}

export class MLPRegressor extends BaseRegressor {
  public model: Sequential;
  public hiddenLayerSizes: number[];
  public optimizerType: OptimizerType;
  public optimizerProps: OptimizerProps;
  public featureSize: number;
  public loss: LossType;
  public activations: ActivationIdentifier[] | ActivationIdentifier;
  public outputSize: number;
  public optimizer: Optimizer;
  public initializer: InitailizerType;
  constructor(params: MLPRegressorParams) {
    super();
    const { hiddenLayerSizes, optimizerProps, optimizerType, activations, loss, initializer } = params;
    this.hiddenLayerSizes = hiddenLayerSizes ? hiddenLayerSizes : null;
    this.optimizerType = optimizerType ? optimizerType : 'adam';
    this.optimizerProps = optimizerProps ? optimizerProps : { learningRate: 0.1 };
    this.optimizer = getOptimizer(this.optimizerType, this.optimizerProps);
    this.activations = activations ? activations : 'sigmoid';
    this.loss = loss ? loss : 'mse';
    this.initializer = initializer ? initializer : 'heNormal';
  }
  public initModelFromWeights(weights: (Float32Array | Int32Array | Uint8Array)[]): void {
    const weightsTensors = weights.map((w) => tensor(w));
    this.initModel(weightsTensors);
  }

  private initModel(weightsTensors: Tensor[] = []): void {
    if (!this.hiddenLayerSizes) {
      throw new TypeError('hidden layer sizes are required before initialize');
    }
    const model = sequential();
    let activations: ActivationIdentifier[];
    if (this.activations) {
      if (typeof(this.activations) === 'string') {
        activations = new Array(this.hiddenLayerSizes.length).fill(this.activations);
      } else if (this.activations instanceof Array) {
        activations = this.activations;
      }
    } else {
      activations = new Array(this.hiddenLayerSizes.length).fill('sigmoid');
    }
    model.add(layers.dense({
      inputShape: [ this.featureSize ],
      units: this.hiddenLayerSizes[0],
      activation: activations[0],
      kernelInitializer: this.initializer,
      biasInitializer: this.initializer
    }));
    for (let i = 0; i < this.hiddenLayerSizes.length; i++) {
      if (i < this.hiddenLayerSizes.length - 1){
        model.add(layers.dense({
          inputShape: [ this.hiddenLayerSizes[i] ],
          units: this.hiddenLayerSizes[i + 1],
          activation: activations[i + 1],
          kernelInitializer: this.initializer,
          biasInitializer: this.initializer
        }));
      }
      if (i === this.hiddenLayerSizes.length - 1) {
        model.add(layers.dense({
          inputShape: [ this.hiddenLayerSizes[i] ],
          units: 1,
          activation: 'linear',
          kernelInitializer: this.initializer,
          biasInitializer: this.initializer
        }));
      }
    }
    model.compile({
      optimizer: this.optimizer,
      loss: getLossFunction(this.loss)
    });
    if (weightsTensors?.length) model.setWeights(weightsTensors);
    this.model = model;
  }

  /** Fit multi-layer perceptron regression model according to X, y. Here we use adam algorithm (a popular gradient-based optimization algorithm) for paramter estimation.
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target values
   * @param params batchSize: batch size: default to 32, epochs: epochs for training, default to Math.ceil(50000 / ( nData / batchSize ))
   */
  public async fit(xData: Tensor | RecursiveArray<number>,
    yData: Tensor | RecursiveArray<number>,
    params: MLPRegressorTrainParams = { batchSize: 32, epochs: -1 }): Promise<void> {

    const { x, y } = this.validateData(xData, yData);
    const nFeature = x.shape[1];
    const nData = x.shape[0];
    const batchSize = params.batchSize ? (nData > params.batchSize ? params.batchSize : nData) : 32;
    const verbose = params.verbose !== false;
    const earlyStopping = params.earlyStopping !== false;
    const maxIterTimes = 50000;
    const defaultEpochs = Math.ceil(maxIterTimes / (nData / batchSize));
    const epochs = params.epochs > 0 ? params.epochs : defaultEpochs;
    const tol = params.tol ? params.tol : 0;
    this.featureSize = nFeature;
    this.initModel();
    const verboseCallback = new CustomCallback({
      onEpochEnd: (epoch, logs) => { console.log(epoch, logs); }
    });
    const earlyStoppingCallback = callbacks.earlyStopping({ monitor: 'loss', minDelta: tol });
    const callbackFuncs = [];
    if (verbose) {
      callbackFuncs.push(verboseCallback);
    }
    if (earlyStopping) {
      callbackFuncs.push(earlyStoppingCallback);
    }
    await this.model.fit(x, y, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: callbackFuncs
    });
  }

  public async predict(xData: Tensor | RecursiveArray<number>): Promise<Tensor> {
    const x = checkArray(xData, 'float32', 2);
    const predY = this.model.predict(x);
    if (predY instanceof Tensor) {
      return squeeze(predY);
    } else {
      return squeeze(stack(predY));
    }
  }

  public getModelWeightsArray(): Promise<RecursiveArray<number>> {
    return Promise.all(this.model.getWeights().map((w: Tensor) => w.array()));
  }

  /**
   * Load model paramters from json string object
   * @param modelJson model json saved as string object
   * @returns model itself
   */
  public async fromJson(modelJson: string): Promise<void> {
    const modelParams = JSON.parse(modelJson);
    if (modelParams.name !== 'MLPRegressor') {
      throw new TypeError(`${modelParams.name} is not MLPRegressor`);
    }
    const {
      optimizerType,
      optimizerProps,
      activations,
      hiddenLayerSizes,
      modelWeights,
      featureSize
    } = modelParams;
    this.featureSize = featureSize;
    this.hiddenLayerSizes = hiddenLayerSizes;
    if (activations) {
      this.activations = activations;
    }
    if (optimizerType && optimizerProps) {
      this.optimizer = getOptimizer(optimizerType, optimizerProps);
      this.optimizerType = optimizerType;
      this.optimizerProps = optimizerProps;
    }
    if (modelWeights?.length) {
      this.initModelFromWeights(modelWeights);
    }
  }

  /**
   * Dump model parameters to json string.
   * @returns Stringfied model parameters
   */
  public async toJson(): Promise<string> {
    const modelParams = {
      name: 'MLPRegressor',
      featureSize: this.featureSize,
      activations: this.activations,
      hiddenLayerSizes: this.hiddenLayerSizes,
      optimizerType: this.optimizerType,
      optimizerProps: this.optimizerProps,
      modelWeights: await this.getModelWeightsArray()
    };
    return JSON.stringify(modelParams);
  }
}
