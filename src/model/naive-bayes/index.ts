import { BaseClassifier } from '../base';
import { Tensor, oneHot, unique, add, sub, log, argMax, cast, squeeze, exp, reshape,
  matMul, transpose, sum, div, booleanMaskAsync, gather, stack, Tensor2D, tensor } from '@tensorflow/tfjs-core';

export type ClassMap = {
  [ key: string ]: number
}

/**
 * Multinomial naive bayes classifier
 */
export class MultinomialNB extends BaseClassifier {

  private conditionProb: Tensor;
  private priorProb: Tensor;
  private classes: Tensor;
  public alpha: number;
  public featureCount: Tensor;
  public classCount: Tensor;
  public classMap: ClassMap;

  constructor (alpha = 1.0) {
    super();
    this.alpha = alpha;
  }

  private updateClassLogPrior() {
    if (this.classCount) {
      const nData = cast(sum(this.classCount), 'float32');
      this.priorProb = sub(log(this.classCount), log(nData));
    }
  }

  private updateFeatureLogProb() {
    if (this.featureCount){
      const axisH = 1;
      const featureCountFixed = add(this.featureCount, this.alpha);
      const classCountFixed = reshape(sum(featureCountFixed, axisH), [ -1, 1 ]);
      this.conditionProb = sub(log(featureCountFixed), log(classCountFixed));
    }
  }

  private updateFeautrCount(featureCount: Tensor) {
    featureCount = cast(featureCount, 'float32');
    if (this.featureCount) {
      this.featureCount = add(this.featureCount, featureCount);
    } else {
      this.featureCount = featureCount;
    }
  }

  private updateClassCount(classCount: Tensor) {
    classCount = cast(classCount, 'float32');
    if (this.classCount) {
      this.classCount = add(this.classCount, classCount);
    } else {
      this.classCount = classCount;
    }
  }

  private getLogLikelihood(X: Tensor2D): Tensor {
    const v = matMul(X, transpose(this.conditionProb));
    const predictions = add(v, this.priorProb);
    return predictions;
  }

  // check if first call to train
  private firstCall(): boolean {
    if (this.featureCount && this.priorProb && this.conditionProb && this.classCount) {
      return false;
    }
    return true;
  }

  // get class map
  private updateClassMap(): void {
    if (this.classes){
      const classData = this.classes.dataSync();
      const classMap: ClassMap = {};
      for (let i = 0; i < classData.length; i++) {
        const key = classData[i];
        classMap[key] = i;
      }
      this.classMap = classMap;
    }
  }

  // get one-hot vector for new input data
  private getNewBatchOneHot(y: Tensor): Tensor {
    const yData = y.dataSync();
    const yInd = yData.map((d: number) => this.classMap[d]);
    return cast(tensor(yInd), 'int32');
  }

  /**
   * Training multinomial naive bayes model according to X, y. Support training multiple batches of data
   * @param xData feature array, two dimension numeric array or 2D Tensor of shape (n_samples, n_features)
   * @param yData label array, one dimension numeric array or 1D Tensor of size n_samples, use different integer values to represent different labels
   * @returns classifier itself
   */
  public async train(xData: Array<any> | Tensor, yData: Array<any> | Tensor): Promise<MultinomialNB> {
    const { x, y } = this.validateData(xData, yData);
    const { values, indices } = unique(y);
    const nClass = values.shape[0];
    const firstCall = this.firstCall();
    let yOneHot;

    if (firstCall) {
      this.classes = values;
      this.updateClassMap();
      yOneHot = oneHot(indices, nClass);
    } else {
      const yInd = this.getNewBatchOneHot(y);
      const nFeatures = x.shape[1];
      if (nFeatures != this.featureCount.shape[1]) {
        throw new Error('feature size does not match to previous training dataset');
      }
      yOneHot = oneHot(yInd, nClass);
    }

    const axisH = 0;
    // update class count
    const classCount = sum(yOneHot, axisH);
    this.updateClassCount(classCount);

    // update feature count
    const featureCounts = [];
    for (let i = 0; i < nClass ; i++) {
      const axisH = 1;
      const axisV = 0;
      const classMask = squeeze(cast(gather(yOneHot, [ i ], axisH), 'bool'));
      const dataI = await booleanMaskAsync(x, classMask);
      const featureCount = sum(dataI, axisV);
      featureCounts.push(featureCount);
    }
    const featureCountTensor = stack(featureCounts);
    this.updateFeautrCount(featureCountTensor);
    // update prior prob
    this.updateClassLogPrior();
    // update feature conditional log prob
    this.updateFeatureLogProb();
    return this;
  }

  /**
   * Perform classification on an array of test vector X.
   * @param x input feature array, two dimension numeric array or 2D Tensor of shape (n_samples, n_features).
   * @returns 1D Tensor of shape n_samples, Predicted target value of X.
   */
  public predict(x: Tensor2D): Tensor {
    const logLikelihood = this.getLogLikelihood(x);
    const axisH = 1;
    const classInd = argMax(logLikelihood, axisH);
    const classVal = gather(this.classes, classInd);
    return classVal;
  }

  /**
   * Return probablity estimates for test vector X.
   * @param x input feature array, two dimension numeric array or 2D Tensor of shape (n_samples, n_features).
   * @returns 2D Tensor of shape (n_samples, n_claases). Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order
   */
  public predictProba(x: Tensor2D): Tensor {
    const axisH = 1;
    const logLikelihood = this.getLogLikelihood(x);
    const likeliHood = exp(logLikelihood);
    const sumLikelihood = reshape(sum(likeliHood, axisH), [ -1, 1 ]);
    const proba = div(likeliHood, sumLikelihood);
    return proba;
  }

  /**
   * Load model parameters from dumped JSON string
   * @param modelJson: JSON string, contains model parameters
   * @returns classifier itself
   */
  public load(modelJson:string): void {
    const modelParams = JSON.parse(modelJson);
    if (modelParams.name != 'MultinomialNB'){
      throw new RangeError(`${modelParams.name} is not a Multinomial Naive Bayes`);
    }
    this.priorProb = modelParams.priorProb ? tensor(modelParams.priorProb) : this.priorProb;
    this.classes = modelParams.classes ? cast(tensor(modelParams.classes), 'int32') : this.classes;
    this.conditionProb = modelParams.conditionProb ? tensor(modelParams.conditionProb) : this.conditionProb;
    this.alpha = modelParams.alpha ? modelParams.alpha : this.alpha;
    this.classCount = modelParams.classCount ? modelParams.classCount : this.classCount;
    this.featureCount = modelParams.featureCount ? modelParams.featureCount : this.featureCount;
  }

  public toJson(): string {
    const modelParams = {
      name: 'MultinomialNB',
      priorProb: this.priorProb.arraySync(),
      conditionProb: this.conditionProb.arraySync(),
      classes: this.classes.arraySync(),
      alpha: this.alpha,
      classCount: this.classCount.arraySync(),
      featureCount: this.featureCount.arraySync()
    };
    return JSON.stringify(modelParams);
  }
}
