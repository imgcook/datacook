import { Tensor, RecursiveArray, matMul, mean, sub, div, sum, slice, divNoNan, sqrt, dispose, tidy, tensor } from '@tensorflow/tfjs-core';
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
    if (!(xData instanceof Tensor)) {
      dispose(xTensor);
    }
  }

  private getCorcovMatrix(xTensor: Tensor, method: PCAMethodType): Tensor {
    return tidy(() => {
      if (method === 'correlation') {
        return getCorrelationMatrix(xTensor);
      }
      return getCovarianceMatrix(xTensor);
    });
  }

  /**
   * This is a private method that will compute covariance matrix and record corresponding results
   * @param xTensor Training data which is a tensor of shape (n_samples, n_features)
   */
  private async fitFull(xTensor: Tensor, nComponents: number, method: PCAMethodType = 'covariance'): Promise<void> {
    const corcovMatrix = this.getCorcovMatrix(xTensor, method);
    const [ eigenValues, eigenVectors ] = await eigenSolve(corcovMatrix, 1e-4, 200, true);
    const nFeatures = eigenValues.shape[0];
    if (this.mean) dispose(this.mean);
    if (this.variance) dispose(this.variance);
    if (this.eigenValues) dispose(this.eigenValues);
    if (this.eigenVectors) dispose(this.eigenVectors);
    if (this.explainedVariance) dispose(this.explainedVariance);
    if (this.explainedVarianceRatio) dispose(this.explainedVarianceRatio);


    this.mean = mean(xTensor, 0);
    this.variance = tidy(() => getVariance(xTensor, 0));
    this.eigenValues = slice(eigenValues, 0, nComponents);
    this.eigenVectors = slice(eigenVectors, [ 0, 0 ], [ nFeatures, nComponents ]);
    this.explainedVariance = this.eigenValues;
    this.explainedVarianceRatio = tidy(() => div(this.eigenValues, sum(eigenValues)));

    // dispose unused tensors
    dispose([ corcovMatrix, eigenVectors, eigenValues ]);
  }

  /**
   * Transform method of PCA
   * @param X Data input for PCA decomposition transform. Tensor of shape (n_samples, n_features)
   * @returns the result after dimension reduction
   */
  public async transform(xData: Tensor | RecursiveArray<number>): Promise<Tensor> {
    return tidy(() => {
      const xTensor = checkArray(xData, 'float32', 2);
      const xCentered = sub(xTensor, this.mean);
      if (this.method == 'covariance'){
        return matMul(xCentered, this.eigenVectors);
      } else {
        return matMul(divNoNan(xCentered, sqrt(this.variance)), this.eigenVectors);
      }
    });
  }

  public async fromJson(modelJson: string): Promise<void> {
    const modelParams = JSON.parse(modelJson);
    if (modelParams.name !== 'PCA') {
      throw new RangeError(`${modelParams.name} is not PCA`);
    }
    const { mean, variance, eigenValues, eigenVectors, explainedVariance, explainedVarianceRatio, method, nComponents } = modelParams;
    this.mean = mean ? tensor(mean) : this.mean;
    this.variance = variance ? tensor(variance) : this.variance;
    this.eigenVectors = eigenVectors ? tensor(eigenVectors) : this.eigenVectors;
    this.eigenValues = eigenValues ? tensor(eigenValues) : this.eigenValues;
    this.explainedVariance = explainedVariance ? tensor(explainedVariance) : this.explainedVariance;
    this.explainedVarianceRatio = explainedVarianceRatio ? tensor(explainedVarianceRatio) : this.explainedVarianceRatio;
    this.method = method ? method : this.method;
    this.nComponents = nComponents ? nComponents : this.nComponents;
  }

  public async toJson(): Promise<string> {
    const modelParams = {
      name: 'PCA',
      mean: this.mean.arraySync(),
      variance: this.variance.arraySync(),
      eigenVectors: this.eigenVectors.arraySync(),
      eigenValues: this.eigenValues.arraySync(),
      explainedVariance: this.explainedVariance.arraySync(),
      explainedVarianceRatio: this.explainedVarianceRatio.arraySync(),
      method: this.method,
      nComponents: this.nComponents
    };
    return JSON.stringify(modelParams);
  }
}
