import { divNoNan, RecursiveArray, sqrt, Tensor, sub, Rank, mul, add, tidy } from '@tensorflow/tfjs-core';
import { getVariance, getMean } from '../stat';
import { checkArray } from '../utils/validation';
import { TransformerMixin } from './base';
export interface StandardScalerParams {
  withMean?: boolean;
  withStd?: boolean;
}

/**
 * Standardize features by removing the mean and scaling to unit variance.
 * The standard score of a sample `x` is calculated as: z = (x - u) / s
 */
export class StandardScaler extends TransformerMixin {
  public mean: Tensor;
  public standardVariance: Tensor;
  public withMean: boolean;
  public withStd: boolean;
  public nFeatures: number;
  public nSamplesSeen: number;

  /**
   * Constuctor of StandardScaler
   * @param params Parameters
   * Option in params:
   * ---
   * `with_mean` : boolean, default=True
   * If True, center the data before scaling
   * This does not work (and will raise an exception) when attempted on
   * sparse matrices, because centering them entails building a dense
   * matrix which in common use cases is likely to be too large to fit in
   * memory.
   * `with_std` : boolean, default=True
   * If True, scale the data to unit variance (or equivalently,
   * unit standard deviation).
   */
  constructor(params?: StandardScalerParams) {
    super();
    if (params?.withMean !== null) {
      this.withMean = params?.withMean !== false;
    } else {
      this.withMean = true;
    }
    if (params?.withStd !== null) {
      this.withStd = params?.withStd !== false;
    } else {
      this.withStd = true;
    }
  }

  /**
   * fit standard scaler
   * @param X input data
   */
  public async fit(X: Tensor | RecursiveArray<number>): Promise<void> {
    const xTensor = checkArray(X, 'float32', 2);
    this.checkAndSetNFeatures(xTensor, true);
    this.mean = getMean(xTensor, 0);
    this.standardVariance = sqrt(getVariance(xTensor, 0));
    this.nFeatures = xTensor.shape[1];
    this.nSamplesSeen = xTensor.shape[0];
  }
  /**
   * transform input to standardlized form
   * @param X input data
   * @returns standardized data
   */
  public async transform(X: Tensor | RecursiveArray<number>): Promise<Tensor> {
    return tidy(() => {
      const xTensor = checkArray(X, 'float32', 2);
      this.checkAndSetNFeatures(xTensor, false);
      const xCentered = this.withMean ? sub(xTensor, this.mean) : xTensor;
      return this.withStd ? divNoNan(xCentered, this.standardVariance) : xCentered;
    });
  }
  /**
   * recover scaled transformed data to original scale
   * @param X input data
   * @returns recovered data
   */
  public async inverseTransform(X: Tensor<Rank> | RecursiveArray<number>): Promise<Tensor> {
    return tidy(() => {
      const xTensor = checkArray(X, 'float32', 2);
      this.checkAndSetNFeatures(xTensor, false);
      const xScaleReturn = this.withStd ? mul(xTensor, this.standardVariance) : xTensor;
      return this.withMean ? add(xScaleReturn, this.mean) : xScaleReturn;
    });
  }
}
