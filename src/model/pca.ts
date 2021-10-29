import { Tensor, RecursiveArray, matMul, mean, sub, div, sum, slice, divNoNan, sqrt } from '@tensorflow/tfjs-core';
import { eigenSolve } from '../linalg';
import { getCovarianceMatrix, getCorrelationMatrix } from '../stat/corcov';
import { checkArray } from '../utils/validation';
import { BaseEstimater } from "./base";
import { getVariance } from '../stat';

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
export class PCA extends BaseEstimater {
  nComponents?: number;
  explainedVariance: Tensor;
  explainedVarianceRatio: Tensor;
  eigenValues: Tensor;
  eigenVectors: Tensor;
  method: PCAMethodType;
  private mean: Tensor;
  private variance: Tensor;
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

  /**
   * Fit the model with input xData
   * @param xData Training data which is a tensor or 2d array of shape (n_samples, n_features)
   */
  public async fit(xData: Tensor | RecursiveArray<number>): Promise<void> {
    const xTensor = checkArray(xData, 'float32', 2);
    if (!this.nComponents) {
      this.nComponents = Math.min(...xTensor.shape) - 1;
    }
    await this.fitFull(xTensor, this.nComponents, this.method);
  }

  private getCorcovMatrix(xTensor: Tensor, method: PCAMethodType): Tensor {
    if (method === 'correlation') {
      return getCorrelationMatrix(xTensor);
    }
    return getCovarianceMatrix(xTensor);
  }

  /**
   * This is a private method that will compute covariance matrix and record corresponding results
   * @param xTensor Training data which is a tensor of shape (n_samples, n_features)
   */
  private async fitFull(xTensor: Tensor, nComponents: number, method: PCAMethodType = 'covariance'): Promise<void> {
    const corcovMatrix = this.getCorcovMatrix(xTensor, method);
    const [ eigenValues, eigenVectors ] = await eigenSolve(corcovMatrix);
    const nFeatures = eigenValues.shape[0];
    this.mean = mean(xTensor, 0);
    this.variance = getVariance(xTensor);
    this.eigenValues = slice(eigenValues, 0, nComponents);
    this.eigenVectors = slice(eigenVectors, [ 0, 0 ], [ nFeatures, nComponents ]);
    this.explainedVariance = this.eigenValues;
    this.explainedVarianceRatio = div(this.eigenValues, sum(eigenValues));
  }

  /**
   * Transform method of PCA
   * @param X Data input for PCA decomposition transform. Tensor of shape (n_samples, n_features)
   * @returns the result after dimension reduction
   */
  public async transform(xData: Tensor | RecursiveArray<number>): Promise<Tensor> {
    const xTensor = checkArray(xData, 'float32', 2);
    const xCentered = sub(xTensor, this.mean);
    if (this.method == 'covariance'){
      return matMul(xCentered, this.eigenVectors);
    } else {
      return matMul( divNoNan(xCentered, sqrt(this.variance)), this.eigenVectors );
    }
  }
}
