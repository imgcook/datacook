import { Matrix, matrix, vector, Vector } from '../core/classes';
import { matMul2d, sub2d, div2d, sqrt1d } from "../core/op";
import { BaseEstimator } from './base';

export type PCAMethodType = 'covariance' | 'correlation';

const defaultPCAMethod = 'covariance';
/**
 * Parameters for PCA
 */
export interface PCAParameters {
  /**
   * The estimated number of components. If not specified,
   * it equals the lesser value of nFeatures and nSamples
   */
  nComponents?: number;
  method?: PCAMethodType;
}

/**
 * PCA
 * Linear dimensionality reduction using Decomposition of the covaraince matrix or corrrelation matrix
 */
export class PCAPredictor extends BaseEstimator {
  nComponents?: number;
  explainedVariance: Vector;
  explainedVarianceRatio: Vector;
  eigenValues: Vector;
  eigenVectors: Matrix;
  method: PCAMethodType;
  private mean: Vector;
  private variance: Vector;
  /**
   * Construction function for PCA
   *
   * Options in params
   * ------------
   * @param params PCA Parameters
   *
   * - `nComponents`: the estimated number of components. If not specified, it equals the lesser value
   * of nFeatures and nSamples
   *
   * - `method`: {'covariance'|'correlation'}. Method used for decomposition. default='covariance'.\
   *     'covariance': use covariance matrix for decomposition.\
   *     'correlation': use correlation matrix for decomposition.
   */
  constructor(params: PCAParameters = {
    method: defaultPCAMethod
  }) {
    super();
    this.nComponents = params.nComponents;
    this.method = params.method ? params.method : defaultPCAMethod;
  }

  public async fromJson(modelJson: string): Promise<void> {
    const modelParams = JSON.parse(modelJson);
    if (modelParams.name !== 'PCA') {
      throw new RangeError(`${modelParams.name} is not PCA`);
    }
    const { mean, variance, eigenValues, eigenVectors, explainedVariance, explainedVarianceRatio, method, nComponents } = modelParams;
    this.mean = mean ? vector(mean) : this.mean;
    this.variance = variance ? vector(variance) : this.variance;
    this.eigenVectors = eigenVectors ? matrix(eigenVectors) : this.eigenVectors;
    this.eigenValues = eigenValues ? vector(eigenValues) : this.eigenValues;
    this.explainedVariance = explainedVariance ? vector(explainedVariance) : this.explainedVariance;
    this.explainedVarianceRatio = explainedVarianceRatio ? vector(explainedVarianceRatio) : this.explainedVarianceRatio;
    this.method = method ? method : this.method;
    this.nComponents = nComponents ? nComponents : this.nComponents;
  }

  /**
   * Transform method of PCA
   * @param X Data input for PCA decomposition transform. Tensor of shape (n_samples, n_features)
   * @returns the result after dimension reduction
   */
  public async transform(xData: number[][]): Promise<number[][]> {
    const xTensor = matrix(xData);
    const xCentered = sub2d(xTensor, this.mean);
    if (this.method == 'covariance'){
      return matMul2d(xCentered, this.eigenVectors).data;
    } else {
      return matMul2d(div2d(xCentered, sqrt1d(this.variance)), this.eigenVectors).data;
    }
  }
}
