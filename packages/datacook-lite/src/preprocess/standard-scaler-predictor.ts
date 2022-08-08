// import { divNoNan, RecursiveArray, sqrt, Tensor, sub, Rank, mul, add, tidy } from '@tensorflow/tfjs-core';

import { matrix, vector, Vector } from '../core/classes';
import { add2d, div2d, mul2d, sub2d } from '../core/op';
import { TransformerMixin } from './base';
export interface StandardScalerParams {
  withMean?: boolean;
  withStd?: boolean;
}

/**
 * Standardize features by removing the mean and scaling to unit variance.
 * The standard score of a sample `x` is calculated as: z = (x - u) / s
 */
export class StandardScalerPredictor extends TransformerMixin {
  public mean: Vector;
  public standardVariance: Vector;
  public withMean: boolean;
  public withStd: boolean;
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

  public async toJson(): Promise<string> {
    const modelParams = {
      mean: this.mean,
      standardVariance: this.standardVariance,
      withMean: this.withMean,
      withStd: this.withStd,
      nFeature: this.nFeature
    };
    return JSON.stringify(modelParams);
  }

  public async fromJson(modelJson: string): Promise<void> {
    const params = JSON.parse(modelJson);
    if (params.name !== 'StandardScaler') {
      throw new RangeError(`${params.name} is not StandardScaler`);
    }
    const { mean, standardVariance, withMean, withStd, nFeature } = params;
    this.mean = mean ? vector(mean) : this.mean;
    this.standardVariance = standardVariance ? vector(standardVariance) : this.standardVariance;
    this.withMean = withMean ? withMean : this.withMean;
    this.withStd = withStd ? withStd : this.withStd;
    this.nFeature = nFeature ? nFeature : this.nFeature;
  }

  /**
   * transform input to standardlized form
   * @param X input data
   * @returns standardized data
   */
  public async transform(X: number[][]): Promise<number[][]> {
    const xTensor = matrix(X);
    this.checkAndSetNFeatures(X, false);
    const xCentered = this.withMean ? sub2d(xTensor, this.mean) : xTensor;
    return this.withStd ? div2d(xCentered, this.standardVariance).data : xCentered.data;
  }
  /**
   * recover scaled transformed data to original scale
   * @param X input data
   * @returns recovered data
   */
  public async inverseTransform(X: number[][]): Promise<number[][]> {
    const xTensor = matrix(X);
    this.checkAndSetNFeatures(X, false);
    const xScaleReturn = this.withStd ? mul2d(xTensor, this.standardVariance) : xTensor;
    return this.withMean ? add2d(xScaleReturn, this.mean).data : xScaleReturn.data;
  }
}
