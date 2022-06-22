import { BaseClassifier } from '../base';
// import { Tensor, add, sub, log, cast, squeeze, exp, reshape, matMul, transpose,
//   sum, booleanMaskAsync, gather, stack, Tensor2D, tensor, divNoNan } from '@tensorflow/tfjs-core';
import { Matrix, Vector, matrix, vector } from '../../core/classes';
import { transpose2d, matMul2d, add2d, exp2d, sum2d, div2d } from '../../core/op';

// import {  }

export type ClassMap = {
  [ key: string ]: number
}

export interface MultinomialNBParam {
  alpha: number
}

/**
 * Multinomial naive bayes classifier
 */
export class MultinomialNBPredictor extends BaseClassifier<string | number> {

  private conditionProb: Matrix;
  private priorProb: Vector;
  public alpha: number;
  public featureCount: Matrix;
  public classCount: Matrix;

  constructor (params: MultinomialNBParam = { alpha: 1.0 }) {
    super();
    this.alpha = params.alpha && params.alpha > 0 ? params.alpha : 1.0;
  }

  private getLogLikelihood(X: Matrix): Matrix {
    const v = matMul2d(X, transpose2d(this.conditionProb));
    const predictions = add2d(v, this.priorProb);
    return predictions;
  }

  /**
   * Perform classification on an array of test vector X.
   * @param x input feature array, two dimension numeric array or 2D Tensor of shape (n_samples, n_features).
   * @returns 1D Tensor of shape n_samples, Predicted target value of X.
   */
  public async predict(X: number[][]): Promise<Array<string | number>> {
    const logLikelihood = this.getLogLikelihood(matrix(X)).data;
    const classVal = await this.classOneHotEncoder.decode(logLikelihood);
    return classVal;
  }

  /**
   * Return probablity estimates for test vector X.
   * @param x input feature array, two dimension numeric array or 2D Tensor of shape (n_samples, n_features).
   * @returns 2D Tensor of shape (n_samples, n_claases). Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order
   */
  public async predictProba(x: number[][]): Promise<number[][]> {
    const axisH = 1;
    const logLikelihood = this.getLogLikelihood(matrix(x));
    const likeliHood = exp2d(logLikelihood);
    const sumLikelihood = sum2d(likeliHood, axisH);
    const proba = div2d(likeliHood, sumLikelihood, axisH);
    return proba.data;
  }

  /**
   * Load model parameters from dumped JSON string
   * @param modelJson: JSON string, contains model parameters
   * @returns classifier itself
   */
  public async load(modelJson:string): Promise<void> {
    const modelParams = JSON.parse(modelJson);
    if (modelParams.name !== 'MultinomialNB'){
      throw new TypeError(`${modelParams.name} is not a Multinomial Naive Bayes`);
    }
    this.priorProb = modelParams.priorProb ? vector(modelParams.priorProb) : this.priorProb;
    modelParams.classes && await this.initClasses(modelParams.classes);
    this.conditionProb = modelParams.conditionProb ? matrix(modelParams.conditionProb) : this.conditionProb;
    this.alpha = modelParams.alpha ? modelParams.alpha : this.alpha;
    this.classCount = modelParams.classCount ? modelParams.classCount : this.classCount;
    this.featureCount = modelParams.featureCount ? matrix(modelParams.featureCount) : this.featureCount;
  }

  public toJson(): string {
    const modelParams = {
      name: 'MultinomialNB',
      priorProb: this.priorProb?.data,
      conditionProb: this.conditionProb?.data,
      classes: this.classes(),
      alpha: this.alpha,
      classCount: this.classCount?.data,
      featureCount: this.featureCount?.data
    };
    return JSON.stringify(modelParams);
  }
}
